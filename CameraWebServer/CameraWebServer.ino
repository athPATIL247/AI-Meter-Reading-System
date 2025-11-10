// ESP32-CAM Quick Test Stream (AI Thinker)
// Upload speed: 115200 (or 921600 if stable)

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ====== WiFi ======
const char* ssid     = "Testing";
const char* password = "12345678";

// ====== AI Thinker ESP32-CAM pin map ======
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// On-board flash LED (AI Thinker)
#define LED_FLASH_PIN      4

WebServer server(80);

// Simple homepage
const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32-CAM</title>
<style>
  body{font-family:system-ui,Arial;margin:20px}
  .row{display:flex;gap:10px;align-items:center;margin-bottom:10px}
  button{padding:.6rem 1rem;border:1px solid #ccc;border-radius:8px;cursor:pointer}
  img,video{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}
  .ip{font-size:.9rem;color:#555}
</style>
</head>
<body>
<h2>ESP32-CAM Quick Test</h2>
<p class="ip">If the image below loads, your camera works. Stream URL: <code>/stream</code></p>
<div class="row">
  <button onclick="fetch('/led?on=1')">LED ON</button>
  <button onclick="fetch('/led?on=0')">LED OFF</button>
  <button onclick="snap()">Snapshot</button>
</div>
<img id="snap" src="/jpg" onerror="this.alt='Snapshot failed';">
<h3>Live Stream</h3>
<img src="/stream" onerror="this.alt='Stream failed';">
<script>
async function snap(){
  const img=document.getElementById('snap');
  img.src='/jpg?ts='+Date.now();
}
</script>
</body>
</html>
)HTML";

// ---------- Handlers ----------
void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleLED() {
  String on = server.hasArg("on") ? server.arg("on") : "0";
  digitalWrite(LED_FLASH_PIN, on == "1" ? HIGH : LOW);
  server.send(200, "text/plain", String("LED ") + (on=="1"?"ON":"OFF"));
}

// Single JPEG snapshot
void handleJPG() {
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb){
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  server.sendHeader("Cache-Control", "no-store");
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  WiFiClient client = server.client();
  client.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// MJPEG stream at /stream
void handleStream() {
  WiFiClient client = server.client();

  String response = 
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
    "Cache-Control: no-store\r\n"
    "Pragma: no-cache\r\n\r\n";
  client.print(response);

  const char* boundary = "--frame";
  for(;;) {
    if (!client.connected()) break;

    camera_fb_t * fb = esp_camera_fb_get();
    if(!fb){
      // try once more after small delay
      delay(30);
      continue;
    }

    client.printf("%s\r\n", boundary);
    client.print("Content-Type: image/jpeg\r\n");
    client.printf("Content-Length: %u\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);

    // pacing for stability
    delay(30);
  }
}

// ---------- Camera init ----------
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size   = FRAMESIZE_VGA;  // good starting size
    config.jpeg_quality = 12;             // 10–15 are fine
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  return (err == ESP_OK);
}

void setup() {
  pinMode(LED_FLASH_PIN, OUTPUT);
  digitalWrite(LED_FLASH_PIN, LOW);

  Serial.begin(115200);
  Serial.setDebugOutput(false);
  delay(500);
  Serial.println("\nBooting…");

  if(!initCamera()){
    Serial.println("Camera init failed");
    // blink LED to indicate error
    for (int i = 0; i < 10; i++) {
  digitalWrite(LED_FLASH_PIN, HIGH);
  delay(150);
  digitalWrite(LED_FLASH_PIN, LOW);
  delay(150);
}
    ESP.restart();
  }
  Serial.println("Camera init OK");

  WiFi.begin(ssid, password);
  Serial.print("WiFi: ");
  Serial.print(ssid);
  Serial.print("  Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/jpg", handleJPG);
  server.on("/stream", handleStream);
  server.on("/led", handleLED);
  server.begin();

  Serial.println("Open http://" + WiFi.localIP().toString() + " in your browser");
  Serial.println("Stream: http://" + WiFi.localIP().toString() + "/stream");
}

void loop() {
  server.handleClient();
}
