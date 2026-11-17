/*
  emg_ble_link.ino -- BLE link tester for two ESP32-C3 boards (also builds for S3/C6).

  Flash the SAME firmware on both boards. The PC script (ble_link_test.py) then tells one board to be the
  TRANSMITTER (BLE peripheral / GATT server that sends notifications, like the implant) and the other the
  RECEIVER (BLE central / GATT client, like the external unit), over each board's own USB serial port.

  Everything is controlled by text lines over USB serial (see README.md). Data from the PC to the transmitter
  is sent as "D <n>\n" followed by n raw bytes.

  Each BLE notification = 4-byte little-endian sequence number + payload (default 240 bytes).
  Sequence 0xFFFFFFFF is the end marker: [seq][total_packets][total_payload_bytes][crc32].

  Arduino IDE settings (Tools menu): Board "ESP32C3 Dev Module" (or your S3/C6), USB CDC On Boot "Enabled".
*/
#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
#include <esp_bt.h>
#include <esp_mac.h>
#include <esp_timer.h>
#include <host/ble_hs.h>
#include <host/ble_gap.h>

#if !defined(CONFIG_NIMBLE_ENABLED)
#error "This sketch needs the NimBLE build of the ESP32 BLE library (ESP32-C3/S3/C6 with Arduino-ESP32 core 3.x)."
#endif
#if !ARDUINO_USB_CDC_ON_BOOT
#warning "Set Tools > USB CDC On Boot > Enabled, otherwise Serial is not the USB port."
#endif

#define LINK Serial
#define FW_VERSION "1.2"  // 1.2 (9 Nov 2026): adds phy=3 (LE Coded S=2, 500 kb/s) and phy=4 (LE Coded S=8, 125 kb/s); 1M/2M unchanged

static const char *SERVICE_UUID = "7d1c0001-2b6a-4a4e-9a65-3b7f2d9c1e01";
static const char *CHAR_UUID = "7d1c0002-2b6a-4a4e-9a65-3b7f2d9c1e01";
static const char *ADV_NAME = "EMGLINK";
static const uint32_t END_MARKER = 0xFFFFFFFFu;

// ------------------------------------------------------------------ configuration
struct Cfg {
  int txpwr = 9;       // dBm
  int phy = 2;         // 1 = 1M, 2 = 2M, 3 = LE Coded S=2 (500 kb/s), 4 = LE Coded S=8 (125 kb/s)
  float ci_ms = 7.5f;  // connection interval requested by the receiver (central)
  int latency = 0;
  int tmo_ms = 2000;   // supervision timeout
  int pay = 240;       // payload bytes per notification
  int mtu = 247;
  bool verify = false;  // receiver checks the synthetic burst pattern
  bool fwd = false;     // receiver forwards every received payload to the PC ("B <seq> <len>\n" + bytes)
  bool cpwr = true;     // set the connection TX power explicitly AFTER connecting (the documented way on ESP32-C3)
} cfg;

enum Role { R_NONE, R_TX, R_RX };
enum Mode { M_STREAM, M_BURST };
static Role role = R_NONE;
static Mode mode = M_STREAM;

// ------------------------------------------------------------------ helpers
static uint32_t crcTable[256];
static void crcInit() {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    crcTable[i] = c;
  }
}
static inline uint32_t crcUpdate(uint32_t crc, const uint8_t *p, size_t n) {  // zlib-compatible, chainable
  crc = ~crc;
  while (n--) crc = crcTable[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
  return ~crc;
}
static inline uint32_t xs32(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}
static void fillPattern(uint8_t *p, size_t n, uint32_t seq) {
  uint32_t s = seq * 2654435761u + 0x9E3779B9u;
  if (!s) s = 1;
  for (size_t i = 0; i < n; i++) {
    s = xs32(s);
    p[i] = (uint8_t)(s >> 24);
  }
}

static const int PWR_DBM[] = {-24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 20};
static const esp_power_level_t PWR_LVL[] = {ESP_PWR_LVL_N24, ESP_PWR_LVL_N21, ESP_PWR_LVL_N18, ESP_PWR_LVL_N15,
                                            ESP_PWR_LVL_N12, ESP_PWR_LVL_N9,  ESP_PWR_LVL_N6,  ESP_PWR_LVL_N3,
                                            ESP_PWR_LVL_N0,  ESP_PWR_LVL_P3,  ESP_PWR_LVL_P6,  ESP_PWR_LVL_P9,
                                            ESP_PWR_LVL_P12, ESP_PWR_LVL_P15, ESP_PWR_LVL_P18, ESP_PWR_LVL_P20};
static const int N_PWR = sizeof(PWR_DBM) / sizeof(PWR_DBM[0]);

static int pwrIndexForDbm(int dbm) {
  int best = 0;
  for (int i = 0; i < N_PWR; i++)
    if (abs(PWR_DBM[i] - dbm) < abs(PWR_DBM[best] - dbm)) best = i;
  return best;
}
static int pwrDbmForLevel(esp_power_level_t lvl) {
  for (int i = 0; i < N_PWR; i++)
    if (PWR_LVL[i] == lvl) return PWR_DBM[i];
  return -999;
}
static void applyTxPower() {
  esp_power_level_t lvl = PWR_LVL[pwrIndexForDbm(cfg.txpwr)];
  int r_def = esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_DEFAULT, lvl);
  int r_adv = esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, lvl);
  int r_scan = esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_SCAN, lvl);
  int r_init = esp_ble_tx_power_set_enhanced(ESP_BLE_ENHANCED_PWR_TYPE_INIT, 0, lvl);
  int r_c0 = esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_CONN_HDL0, lvl);  // documented as "only after a connection exists"
  LINK.printf("EV TXPWR requested=%d applied=%d conn=%d rc_default=%d rc_adv=%d rc_scan=%d rc_init=%d rc_conn0_before_connect=%d\n",
              cfg.txpwr, pwrDbmForLevel(lvl), pwrDbmForLevel(esp_ble_tx_power_get(ESP_BLE_PWR_TYPE_CONN_HDL0)), r_def, r_adv,
              r_scan, r_init, r_c0);
}

// ------------------------------------------------------------------ shared link state
static BLEServer *pServer = nullptr;
static BLECharacteristic *pChar = nullptr;
static BLEClient *pClient = nullptr;
static BLERemoteCharacteristic *pRemoteChar = nullptr;
static BLEAdvertisedDevice *foundDev = nullptr;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

static volatile bool connected = false;
static volatile bool subscribed = false;
static volatile uint16_t connHandle = 0xFFFF;
static volatile bool advWanted = false;
static volatile bool scanWanted = false;
static volatile bool doConnect = false;
static volatile bool connecting = false;   // true while rxConnect() is using the found device
static volatile bool needLinkSetup = false;
static uint32_t linkSetupAtMs = 0;

// ------------------------------------------------------------------ transmitter state
static const size_t RING = 24576;
static uint8_t ring[RING];
static uint32_t wrTotal = 0, rdTotal = 0;  // monotonic byte counters
static uint32_t txSeq = 0, txPkts = 0, txBytes = 0, txCrc = 0, txRetries = 0, txErrors = 0, ringDrops = 0;
static bool txActive = false, txFin = false, endPending = false;
static uint32_t burstDurMs = 0, burstPkts = 0, burstRate = 0;
static int64_t txStartUs = 0, txEndUs = 0, nextSendUs = 0;
static uint8_t txbuf[4 + 256];
static uint32_t lastKSent = 0xFFFFFFFFu;
static uint32_t lastKMs = 0;

// ------------------------------------------------------------------ receiver state
static volatile uint32_t rxPkts = 0, rxBytes = 0, rxLost = 0, rxDup = 0, rxOoo = 0, rxBad = 0, rxExpect = 0;
static volatile int64_t rxFirstUs = 0, rxLastUs = 0, rxMaxGapUs = 0;
static volatile uint32_t rxHist[7];
static volatile bool rxEnded = false, rxReported = false;
static uint32_t rxCrc = 0;
static volatile uint32_t endPkts = 0, endBytes = 0, endCrc = 0;
static const int64_t HIST_US[6] = {1000, 5000, 10000, 20000, 50000, 100000};

// forwarding of received payloads to the PC: filled in the BLE callback, drained in loop()
struct FwdSlot {
  uint32_t seq;
  uint16_t len;
  uint8_t data[244];
};
static const int FWD_SLOTS = 64;
static FwdSlot fwdq[FWD_SLOTS];
static volatile uint32_t fwdHead = 0, fwdTail = 0, fwdDrops = 0;

// ------------------------------------------------------------------ reporting
static uint32_t lastSecMs = 0, secPkts = 0, secBytes = 0, secLost = 0;
static uint32_t monMs = 0, lastMonMs = 0;  // "MON <ms>": print "R <ms> <rssi>" this often while connected (0 = off)

// raw readback of the level index the controller reports for each state (0 = -24 dBm ... 8 = 0 dBm, 11 = +9, 15 = +20, 255 = invalid)
static void printPwrDebug(const char *tag) {
  uint16_t h = connected ? connHandle : 0;
  LINK.printf("EV PWRDBG %s default=%d adv=%d scan=%d init=%d conn=%d legacy_default=%d conn_handle=%u\n", tag,
              (int)esp_ble_tx_power_get_enhanced(ESP_BLE_ENHANCED_PWR_TYPE_DEFAULT, 0),
              (int)esp_ble_tx_power_get_enhanced(ESP_BLE_ENHANCED_PWR_TYPE_ADV, 0),
              (int)esp_ble_tx_power_get_enhanced(ESP_BLE_ENHANCED_PWR_TYPE_SCAN, 0),
              (int)esp_ble_tx_power_get_enhanced(ESP_BLE_ENHANCED_PWR_TYPE_INIT, 0),
              (int)esp_ble_tx_power_get_enhanced(ESP_BLE_ENHANCED_PWR_TYPE_CONN, h),
              (int)esp_ble_tx_power_get(ESP_BLE_PWR_TYPE_DEFAULT), (unsigned)h);
}

static void printParams() {
  if (!connected) {
    LINK.println("EV PARAMS connected=0");
    return;
  }
  ble_gap_conn_desc desc;
  uint8_t txp = 0, rxp = 0;
  int8_t rssi = 0;
  int rc1 = ble_gap_conn_find(connHandle, &desc);
  int rc2 = ble_gap_read_le_phy(connHandle, &txp, &rxp);
  int rc3 = ble_gap_conn_rssi(connHandle, &rssi);
  uint16_t mtu = (role == R_RX && pClient) ? pClient->getMTU() : (pServer ? pServer->getPeerMTU(connHandle) : 0);
  LINK.printf("EV PARAMS role=%s itvl_ms=%.2f latency=%d timeout_ms=%d phy_tx=%d phy_rx=%d mtu=%u rssi=%d ok=%d%d%d\n",
              role == R_TX ? "TX" : "RX", rc1 == 0 ? desc.conn_itvl * 1.25f : -1.0f, rc1 == 0 ? desc.conn_latency : -1,
              rc1 == 0 ? desc.supervision_timeout * 10 : -1, txp, rxp, mtu, rc3 == 0 ? rssi : 0, rc1 == 0, rc2 == 0,
              rc3 == 0);
}

static void applyLinkSetup() {  // PHY + data length (both roles may ask; the controller keeps the best common one)
  if (!connected) return;
  uint8_t mask = (cfg.phy == 2) ? BLE_GAP_LE_PHY_2M_MASK : (cfg.phy >= 3 ? BLE_GAP_LE_PHY_CODED_MASK : BLE_GAP_LE_PHY_1M_MASK);
  uint16_t opts = (cfg.phy == 3) ? BLE_GAP_LE_PHY_CODED_S2 : (cfg.phy == 4 ? BLE_GAP_LE_PHY_CODED_S8 : BLE_GAP_LE_PHY_CODED_ANY);
  int rc = ble_gap_set_prefered_le_phy(connHandle, mask, mask, opts);
  // 1M/2M: 2120 us is the longest data packet on the 1M PHY. Coded: a 251-byte packet lasts up to 17040 us (S=8), so allow that.
  int rd = ble_gap_set_data_len(connHandle, 251, cfg.phy >= 3 ? 17040 : 2120);
  LINK.printf("EV LINKSETUP phy_req=%d phy_opts=%u phy_rc=%d dle_rc=%d\n", cfg.phy, (unsigned)opts, rc, rd);
  printPwrDebug("before_conn_power_set");
  if (cfg.cpwr) {  // connection TX power can only be set once the connection exists
    int rp = esp_ble_tx_power_set_enhanced(ESP_BLE_ENHANCED_PWR_TYPE_CONN, connHandle, PWR_LVL[pwrIndexForDbm(cfg.txpwr)]);
    LINK.printf("EV CONNPWR requested=%d rc=%d\n", cfg.txpwr, rp);
    printPwrDebug("after_conn_power_set");
  }
}

static void printTxStat() {
  uint32_t dur_ms = (uint32_t)((txEndUs - txStartUs) / 1000);
  LINK.printf("TXSTAT mode=%s pkts=%u bytes=%u crc=%08x retries=%u errors=%u ring_drops=%u dur_ms=%u txpwr=%d pay=%d\n",
              mode == M_BURST ? "burst" : "stream", txPkts, txBytes, txCrc, txRetries, txErrors, ringDrops, dur_ms,
              cfg.txpwr, cfg.pay);
}

static void printRxStat(const char *why) {
  uint32_t pk, by, lost, dup, ooo, bad;
  int64_t f, l, mg;
  uint32_t h[7];
  portENTER_CRITICAL(&mux);
  pk = rxPkts; by = rxBytes; lost = rxLost; dup = rxDup; ooo = rxOoo; bad = rxBad; f = rxFirstUs; l = rxLastUs; mg = rxMaxGapUs;
  for (int i = 0; i < 7; i++) h[i] = rxHist[i];
  portEXIT_CRITICAL(&mux);
  LINK.printf("RXSTAT why=%s pkts=%u bytes=%u lost=%u dup=%u ooo=%u bad=%u crc=%08x dur_us=%u max_gap_us=%u "
              "end_pkts=%u end_bytes=%u end_crc=%08x fwd_drops=%u hist=%u,%u,%u,%u,%u,%u,%u\n",
              why, pk, by, lost, dup, ooo, bad, rxCrc, (uint32_t)(l - f), (uint32_t)mg, endPkts, endBytes, endCrc, fwdDrops,
              h[0], h[1], h[2], h[3], h[4], h[5], h[6]);
}

// ------------------------------------------------------------------ BLE callbacks (transmitter side)
class SrvCb : public BLEServerCallbacks {
  void onConnect(BLEServer *s, ble_gap_conn_desc *desc) override {
    connHandle = desc->conn_handle;
    connected = true;
    subscribed = false;
    needLinkSetup = true;
    linkSetupAtMs = millis() + 400;
    LINK.printf("EV CONNECTED role=TX handle=%u\n", connHandle);
  }
  void onDisconnect(BLEServer *s, ble_gap_conn_desc *desc) override {
    connected = false;
    subscribed = false;
    LINK.println("EV DISCONNECTED role=TX");
  }
  void onMtuChanged(BLEServer *s, ble_gap_conn_desc *desc, uint16_t mtu) override { LINK.printf("EV MTU %u\n", mtu); }
};
class ChrCb : public BLECharacteristicCallbacks {
  void onSubscribe(BLECharacteristic *c, ble_gap_conn_desc *desc, uint16_t subValue) override {
    subscribed = (subValue & 1) != 0;
    LINK.printf("EV SUBSCRIBED %d\n", (int)subscribed);
  }
};

// ------------------------------------------------------------------ BLE callbacks (receiver side)
static void onNotify(BLERemoteCharacteristic *c, uint8_t *d, size_t n, bool isNotify) {
  int64_t now = esp_timer_get_time();
  if (n < 4) return;
  uint32_t seq;
  memcpy(&seq, d, 4);
  if (seq == END_MARKER) {
    portENTER_CRITICAL(&mux);
    if (n >= 16) {
      memcpy((void *)&endPkts, d + 4, 4);
      memcpy((void *)&endBytes, d + 8, 4);
      memcpy((void *)&endCrc, d + 12, 4);
    }
    if (endPkts > rxExpect) rxLost += endPkts - rxExpect;  // packets lost right before the marker
    rxEnded = true;
    portEXIT_CRITICAL(&mux);
    return;
  }
  size_t len = n - 4;
  bool bad = false;
  if (cfg.verify && len <= 256) {
    uint8_t ref[256];
    fillPattern(ref, len, seq);
    bad = memcmp(ref, d + 4, len) != 0;
  }
  rxCrc = crcUpdate(rxCrc, d + 4, len);
  if (cfg.fwd) {
    if (fwdHead - fwdTail < FWD_SLOTS && len <= sizeof(fwdq[0].data)) {
      FwdSlot *s = &fwdq[fwdHead % FWD_SLOTS];
      s->seq = seq;
      s->len = (uint16_t)len;
      memcpy(s->data, d + 4, len);
      __sync_synchronize();
      fwdHead++;
    } else {
      fwdDrops++;
    }
  }
  portENTER_CRITICAL(&mux);
  if (rxPkts == 0) {
    rxFirstUs = now;
  } else {
    int64_t gap = now - rxLastUs;
    int b = 6;
    for (int i = 0; i < 6; i++)
      if (gap < HIST_US[i]) { b = i; break; }
    rxHist[b]++;
    if (gap > rxMaxGapUs) rxMaxGapUs = gap;
  }
  rxLastUs = now;
  if (seq == rxExpect) {
    rxExpect = seq + 1;
  } else if (seq > rxExpect) {
    rxLost += seq - rxExpect;
    rxExpect = seq + 1;
  } else if (seq + 1 == rxExpect) {
    rxDup++;
  } else {
    rxOoo++;
  }
  rxPkts++;
  rxBytes += len;
  if (bad) rxBad++;
  portEXIT_CRITICAL(&mux);
}

class ClientCb : public BLEClientCallbacks {
  void onConnect(BLEClient *c) override {}
  void onDisconnect(BLEClient *c) override {
    connected = false;
    subscribed = false;
    LINK.println("EV DISCONNECTED role=RX");
  }
};
// Scanning is done in short windows and the library's result list is cleared after each one. The library stores every
// advertiser it hears for the whole scan, and the room is full of phones and earbuds: one 30 s scan, repeated, exhausted
// the heap and the board crashed (abort in BLEAdvertisedDevice::parseAdvertisement, found by link calibration, 3 Nov 2026).
static uint32_t scanBeginMs = 0;
static volatile bool scanRestart = false;
static const uint32_t SCAN_WINDOW_S = 2;
static const uint32_t SCAN_TOTAL_MS = 30000;

class AdvCb : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice dev) override {
    if (doConnect || connected || connecting) return;  // ignore anything heard after we already have our device
    if (dev.haveServiceUUID() && dev.isAdvertisingService(BLEUUID(SERVICE_UUID))) {
      LINK.printf("EV FOUND rssi=%d\n", dev.getRSSI());
      if (foundDev) {
        delete foundDev;
        foundDev = nullptr;
      }
      foundDev = new BLEAdvertisedDevice(dev);
      doConnect = true;                    // set BEFORE stop(): stop() calls onScanDone synchronously
      BLEDevice::getScan()->stop();
    }
  }
};
static AdvCb advCb;  // one object for the whole run (it used to be allocated again on every SCAN command)

static void onScanDone(BLEScanResults r) {
  BLEDevice::getScan()->clearResults();
  if (scanWanted && !doConnect && !connected) {
    if (millis() - scanBeginMs > SCAN_TOTAL_MS) {
      LINK.println("EV SCAN_TIMEOUT");
      scanWanted = false;
    } else {
      scanRestart = true;                  // restarted from loop(), not from inside this callback
    }
  }
}

static void startScanWindow() {
  BLEScan *sc = BLEDevice::getScan();
  sc->clearResults();
  sc->setAdvertisedDeviceCallbacks(&advCb);
  sc->setActiveScan(true);
  sc->setInterval(100);
  sc->setWindow(80);
  sc->start(SCAN_WINDOW_S, onScanDone, false);
}

static bool rxConnectTo(BLEAdvertisedDevice *dev) {
  if (!pClient) {
    pClient = BLEDevice::createClient();
    pClient->setClientCallbacks(new ClientCb());
  }
  if (!pClient->connect(dev)) {
    LINK.println("EV CONNECT_FAILED");
    return false;
  }
  connHandle = pClient->getConnId();
  connected = true;
  BLERemoteService *svc = pClient->getService(BLEUUID(SERVICE_UUID));
  pRemoteChar = svc ? svc->getCharacteristic(BLEUUID(CHAR_UUID)) : nullptr;
  if (!pRemoteChar || !pRemoteChar->canNotify()) {
    LINK.println("EV SERVICE_NOT_FOUND");
    pClient->disconnect();
    return false;
  }
  uint16_t itvl = (uint16_t)(cfg.ci_ms / 1.25f + 0.5f);
  pClient->updateConnParams(itvl, itvl, cfg.latency, cfg.tmo_ms / 10);
  pRemoteChar->registerForNotify(onNotify);
  subscribed = true;
  LINK.printf("EV CONNECTED role=RX handle=%u\n", connHandle);
  needLinkSetup = true;
  linkSetupAtMs = millis() + 400;
  return true;
}

static bool rxConnect() {  // takes ownership of the found device and always frees it
  BLEAdvertisedDevice *dev = foundDev;
  foundDev = nullptr;
  if (!dev) return false;
  connecting = true;
  bool ok = rxConnectTo(dev);
  connecting = false;
  delete dev;
  return ok;
}

// ------------------------------------------------------------------ transmitter pump
static int sendPacket(const uint8_t *data, size_t len) {
  os_mbuf *om = ble_hs_mbuf_from_flat(data, len);
  if (!om) return BLE_HS_ENOMEM;
  return ble_gatts_notify_custom(connHandle, pChar->getHandle(), om);
}

static void ringRead(uint8_t *dst, size_t n) {
  size_t pos = rdTotal % RING;
  size_t first = min(n, RING - pos);
  memcpy(dst, ring + pos, first);
  if (n > first) memcpy(dst + first, ring, n - first);
}

static void txPump() {
  if (!txActive || !connected || !subscribed) return;
  static int failStreak = 0;
  if (endPending) {
    uint8_t m[16];
    uint32_t mk = END_MARKER;
    memcpy(m, &mk, 4);
    memcpy(m + 4, &txPkts, 4);
    memcpy(m + 8, &txBytes, 4);
    memcpy(m + 12, &txCrc, 4);
    if (sendPacket(m, 16) == 0) {
      endPending = false;
      txActive = false;
      printTxStat();
    } else {
      txRetries++;
      delay(1);
    }
    return;
  }
  int maxPay = min(cfg.pay, 244);
  for (int budget = 0; budget < 16; budget++) {
    int64_t now = esp_timer_get_time();
    size_t len;
    if (mode == M_BURST) {
      bool timeUp = burstDurMs && txStartUs && (now - txStartUs) >= (int64_t)burstDurMs * 1000;
      bool pktUp = burstPkts && txSeq >= burstPkts;
      if (timeUp || pktUp) {
        txEndUs = now;
        endPending = true;
        return;
      }
      if (burstRate && nextSendUs && now < nextSendUs) return;
      len = maxPay;
      fillPattern(txbuf + 4, len, txSeq);
    } else {
      uint32_t avail = wrTotal - rdTotal;
      if (avail >= (uint32_t)maxPay) len = maxPay;
      else if (txFin && avail > 0) len = avail;
      else if (txFin && avail == 0) {
        txEndUs = now;
        endPending = true;
        return;
      } else return;
      ringRead(txbuf + 4, len);
    }
    memcpy(txbuf, &txSeq, 4);
    int rc = sendPacket(txbuf, 4 + len);
    if (rc == 0) {
      failStreak = 0;
      if (txStartUs == 0) {
        txStartUs = now;
        nextSendUs = now;
      }
      txCrc = crcUpdate(txCrc, txbuf + 4, len);
      txBytes += len;
      txPkts++;
      txSeq++;
      txEndUs = now;
      if (mode == M_STREAM) rdTotal += len;
      if (burstRate) nextSendUs += (int64_t)len * 8000000LL / burstRate;
    } else {
      if (rc == BLE_HS_ENOTCONN) connected = false;
      if (rc == BLE_HS_ENOMEM || rc == BLE_HS_EBUSY) txRetries++;
      else txErrors++;
      if (++failStreak > 3) delay(1);
      return;
    }
  }
}

// ------------------------------------------------------------------ command handling
static char lineBuf[160];
static size_t lineLen = 0;
static size_t binRemaining = 0;

static void clearCounters() {
  portENTER_CRITICAL(&mux);
  rxPkts = rxBytes = rxLost = rxDup = rxOoo = rxBad = rxExpect = 0;
  rxFirstUs = rxLastUs = rxMaxGapUs = 0;
  for (int i = 0; i < 7; i++) rxHist[i] = 0;
  rxEnded = rxReported = false;
  endPkts = endBytes = endCrc = 0;
  portEXIT_CRITICAL(&mux);
  rxCrc = 0;
  fwdHead = fwdTail = fwdDrops = 0;
  wrTotal = rdTotal = 0;
  txSeq = txPkts = txBytes = txCrc = txRetries = txErrors = ringDrops = 0;
  txActive = txFin = endPending = false;
  txStartUs = txEndUs = nextSendUs = 0;
  lastKSent = 0xFFFFFFFFu;
  lastSecMs = millis();
  secPkts = secBytes = secLost = 0;
}

static long argLong(const char *s, const char *key, long def) {
  const char *p = strstr(s, key);
  if (!p) return def;
  return strtol(p + strlen(key), nullptr, 10);
}
static float argFloat(const char *s, const char *key, float def) {
  const char *p = strstr(s, key);
  if (!p) return def;
  return strtof(p + strlen(key), nullptr);
}

static void handleLine(char *l) {
  if (!strcmp(l, "PING")) {
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_BT);
    LINK.printf("PONG fw=%s chip=%s mac=%02x:%02x:%02x:%02x:%02x:%02x role=%d heap=%u heap_min=%u\n", FW_VERSION, ESP.getChipModel(),
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], (int)role, (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMinFreeHeap());
  } else if (!strncmp(l, "ROLE ", 5)) {
    role = !strcmp(l + 5, "TX") ? R_TX : (!strcmp(l + 5, "RX") ? R_RX : R_NONE);
    LINK.printf("OK ROLE %s\n", l + 5);
  } else if (!strncmp(l, "SET ", 4)) {
    cfg.txpwr = argLong(l, "txpwr=", cfg.txpwr);
    cfg.phy = argLong(l, "phy=", cfg.phy);
    cfg.ci_ms = argFloat(l, "ci=", cfg.ci_ms);
    cfg.latency = argLong(l, "lat=", cfg.latency);
    cfg.tmo_ms = argLong(l, "tmo=", cfg.tmo_ms);
    cfg.pay = constrain(argLong(l, "pay=", cfg.pay), 1, 244);
    cfg.verify = argLong(l, "verify=", cfg.verify) != 0;
    cfg.fwd = argLong(l, "fwd=", cfg.fwd) != 0;
    cfg.cpwr = argLong(l, "cpwr=", cfg.cpwr) != 0;
    applyTxPower();
    LINK.printf("OK SET txpwr=%d phy=%d ci=%.2f lat=%d tmo=%d pay=%d verify=%d fwd=%d\n", cfg.txpwr, cfg.phy, cfg.ci_ms,
                cfg.latency, cfg.tmo_ms, cfg.pay, (int)cfg.verify, (int)cfg.fwd);
  } else if (!strcmp(l, "ADV")) {
    advWanted = true;
    BLEDevice::startAdvertising();
    LINK.println("OK ADV");
  } else if (!strcmp(l, "SCAN")) {
    scanWanted = true;
    doConnect = false;
    scanBeginMs = millis();
    startScanWindow();
    LINK.printf("OK SCAN heap=%u\n", (unsigned)ESP.getFreeHeap());
  } else if (!strcmp(l, "STOP")) {
    advWanted = scanWanted = false;
    scanRestart = false;
    BLEDevice::getAdvertising()->stop();
    BLEDevice::getScan()->stop();
    BLEDevice::getScan()->clearResults();
    if (connected) ble_gap_terminate(connHandle, BLE_ERR_REM_USER_CONN_TERM);
    txActive = false;
    LINK.println("OK STOP");
  } else if (!strcmp(l, "CLEAR")) {
    clearCounters();
    LINK.println("OK CLEAR");
  } else if (!strncmp(l, "MON ", 4)) {
    monMs = (uint32_t)strtoul(l + 4, nullptr, 10);
    if (monMs && monMs < 50) monMs = 50;  // the RSSI read is a blocking controller command
    LINK.printf("OK MON %u\n", monMs);
  } else if (!strcmp(l, "PWR?")) {
    printPwrDebug("query");
  } else if (!strcmp(l, "INFO")) {
    printParams();
  } else if (!strcmp(l, "STATS")) {
    printRxStat("query");
    printTxStat();
  } else if (!strncmp(l, "BEGIN ", 6)) {
    clearCounters();
    mode = strstr(l, "burst") ? M_BURST : M_STREAM;
    burstDurMs = argLong(l, "dur_ms=", 0);
    burstPkts = argLong(l, "pkts=", 0);
    burstRate = argLong(l, "rate=", 0);
    txActive = true;
    LINK.printf("OK BEGIN %s\n", mode == M_BURST ? "burst" : "stream");
  } else if (!strcmp(l, "FIN")) {
    txFin = true;
    LINK.println("OK FIN");
  } else if (!strncmp(l, "D ", 2)) {
    binRemaining = (size_t)strtoul(l + 2, nullptr, 10);
  } else {
    LINK.printf("ERR unknown '%s'\n", l);
  }
}

static void pollSerial() {
  int guard = 0;
  while (LINK.available() && guard++ < 4096) {
    if (binRemaining > 0) {
      uint8_t tmp[256];
      size_t want = min(binRemaining, sizeof(tmp));
      size_t got = LINK.read(tmp, min((size_t)LINK.available(), want));
      for (size_t i = 0; i < got; i++) {
        if (wrTotal - rdTotal < RING) ring[wrTotal++ % RING] = tmp[i];
        else ringDrops++;
      }
      binRemaining -= got;
    } else {
      int c = LINK.read();
      if (c == '\n') {
        lineBuf[lineLen] = 0;
        if (lineLen) handleLine(lineBuf);
        lineLen = 0;
      } else if (c != '\r' && c >= 0 && lineLen < sizeof(lineBuf) - 1) {
        lineBuf[lineLen++] = (char)c;
      }
    }
  }
}

// ------------------------------------------------------------------ setup / loop
void setup() {
  crcInit();
  LINK.setRxBufferSize(16384);
  LINK.begin(115200);
  delay(300);
  BLEDevice::init(ADV_NAME);
  BLEDevice::setMTU(cfg.mtu);
  applyTxPower();

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new SrvCb());
  BLEService *svc = pServer->createService(SERVICE_UUID);
  pChar = svc->createCharacteristic(CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  pChar->setCallbacks(new ChrCb());
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);

  uint32_t t = crcUpdate(0, (const uint8_t *)"123456789", 9);
  LINK.printf("READY fw=%s chip=%s crc_selftest=%s\n", FW_VERSION, ESP.getChipModel(), t == 0xCBF43926u ? "OK" : "BAD");
}

void loop() {
  pollSerial();

  if (scanRestart && scanWanted && !doConnect && !connected) {
    scanRestart = false;
    startScanWindow();
  }
  if (doConnect && role == R_RX) {
    doConnect = false;
    if (rxConnect()) scanWanted = false;
  }
  if (advWanted && role == R_TX && !connected) {
    static uint32_t lastAdv = 0;
    if (millis() - lastAdv > 1500) {  // (re)start advertising after a disconnect
      lastAdv = millis();
      BLEDevice::startAdvertising();
    }
  }
  if (needLinkSetup && connected && millis() > linkSetupAtMs) {
    needLinkSetup = false;
    applyLinkSetup();
    delay(700);
    printParams();
  }

  if (role == R_TX) {
    txPump();
    if (millis() - lastKMs >= 20 && rdTotal != lastKSent) {
      lastKMs = millis();
      lastKSent = rdTotal;
      LINK.printf("K %u\n", rdTotal);
    }
  }

  if (connected && monMs && millis() - lastMonMs >= monMs) {
    lastMonMs = millis();
    int8_t r = 0;
    if (ble_gap_conn_rssi(connHandle, &r) == 0) LINK.printf("R %u %d\n", millis(), r);
  }

  if (connected && millis() - lastSecMs >= 1000) {
    lastSecMs = millis();
    int8_t rssi = 0;
    int rc = ble_gap_conn_rssi(connHandle, &rssi);
    if (role == R_RX) {
      uint32_t pk = rxPkts, by = rxBytes, lo = rxLost;
      LINK.printf("T ms=%u pkts=%u bytes=%u lost=%u rssi=%d\n", millis(), pk - secPkts, by - secBytes, lo - secLost,
                  rc == 0 ? rssi : 0);
      secPkts = pk; secBytes = by; secLost = lo;
    } else if (role == R_TX) {
      LINK.printf("S ms=%u pkts=%u bytes=%u retries=%u rssi=%d\n", millis(), txPkts, txBytes, txRetries, rc == 0 ? rssi : 0);
    }
  }

  if (role == R_RX && cfg.fwd) {  // stream the received payloads to the PC, in arrival order
    static uint8_t out[300];
    for (int n = 0; n < 8 && fwdTail != fwdHead; n++) {
      FwdSlot *s = &fwdq[fwdTail % FWD_SLOTS];
      int h = snprintf((char *)out, 32, "B %u %u\n", (unsigned)s->seq, (unsigned)s->len);
      memcpy(out + h, s->data, s->len);
      LINK.write(out, h + s->len);
      fwdTail++;
    }
  }

  if (role == R_RX && !rxReported && fwdTail == fwdHead) {  // report only after everything was forwarded
    bool ended = rxEnded;
    bool idle = rxPkts > 0 && (esp_timer_get_time() - rxLastUs) > 3000000;
    if (ended || idle) {
      rxReported = true;
      printRxStat(ended ? "end" : "timeout");
    }
  }
  delay(0);
}
