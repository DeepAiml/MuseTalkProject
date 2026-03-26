using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

public class MuseWsConnection : MonoBehaviour
{
    private const int SAMPLE_RATE = 16000;
    private const int CHUNK_SIZE = 512;
    private const string WS_URL = "wss://backend.datavivservers.in/ws";
    private const float AUDIO_INTERVAL = 0.02f;
    private const int MAX_RECONNECT_ATTEMPTS = 10;
    private const float RECONNECT_BASE_DELAY = 1f;
    private const float RECONNECT_MAX_DELAY = 30f;

    [Header("UI")]
    public Button micBtnImage;

    [Header("State")]
    public bool micOn = false;
    public string token = "";
    public string session_Id = "";

    private ClientWebSocket ws;
    private CancellationTokenSource cts;
    private bool isConnecting = false;
    private int reconnectAttempts = 0;

    private AudioClip micClip;
    private string micDevice;
    private int lastSamplePosition = 0;
    private List<float> micBuffer = new List<float>();
    private Coroutine micCoroutine;

    async void Start()
    {
        await ConnectNative();
        UpdateMicUI(false);
    }

    #region Connection

    private async Task ConnectNative()
    {
        if (isConnecting) return;
        isConnecting = true;

        cts?.Dispose();
        cts = new CancellationTokenSource();
        ws = new ClientWebSocket();

        try
        {
            await ws.ConnectAsync(new Uri(WS_URL), cts.Token);
            reconnectAttempts = 0;
            Debug.Log("[WS] Connected");
            _ = ReceiveLoop();
        }
        catch (Exception e)
        {
            Debug.LogError($"[WS] Connection failed: {e.Message}");
            StartCoroutine(ReconnectAfterDelay());
        }
        finally
        {
            isConnecting = false;
        }
    }

    private IEnumerator ReconnectAfterDelay()
    {
        if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS)
        {
            Debug.LogError("[WS] Max reconnection attempts reached");
            yield break;
        }

        reconnectAttempts++;
        float delay = Mathf.Min(
            RECONNECT_BASE_DELAY * Mathf.Pow(2, reconnectAttempts - 1),
            RECONNECT_MAX_DELAY
        );
        Debug.Log($"[WS] Reconnecting in {delay:F1}s (attempt {reconnectAttempts}/{MAX_RECONNECT_ATTEMPTS})");
        yield return new WaitForSeconds(delay);
        _ = ConnectNative();
    }

    private async Task ReceiveLoop()
    {
        var buffer = new byte[65536];
        var segment = new ArraySegment<byte>(buffer);

        try
        {
            while (ws != null && ws.State == WebSocketState.Open)
            {
                using (var ms = new MemoryStream())
                {
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await ws.ReceiveAsync(segment, cts.Token);
                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            Debug.Log("[WS] Server closed connection");
                            StartCoroutine(ReconnectAfterDelay());
                            return;
                        }
                        ms.Write(buffer, 0, result.Count);
                    } while (!result.EndOfMessage);

                    if (result.MessageType == WebSocketMessageType.Binary)
                    {
                        byte[] data = ms.ToArray();
                        if (data.Length > 0 && ImageConvert.Instance != null)
                        {
                            ImageConvert.Instance.OnBinaryMessage(data);
                        }
                    }
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (WebSocketException e)
        {
            Debug.LogError($"[WS] WebSocket error: {e.Message}");
            StartCoroutine(ReconnectAfterDelay());
        }
        catch (Exception e)
        {
            Debug.LogError($"[WS] Receive error: {e.Message}");
            StartCoroutine(ReconnectAfterDelay());
        }
    }

    private bool IsConnected()
    {
        return ws != null && ws.State == WebSocketState.Open;
    }

    #endregion

    #region Token

    public void UpdateToken(string newToken, string sessionId, float delay)
    {
        StartCoroutine(SendTokenAfterDelay(newToken, sessionId, delay));
    }

    private IEnumerator SendTokenAfterDelay(string newToken, string sessionId, float delay)
    {
        yield return new WaitForSeconds(delay);

        var msg = JsonUtility.ToJson(new WsTextMessage
        {
            type = "submit_token",
            token = newToken,
            session_id = sessionId
        });
        SendText(msg);
    }

    #endregion

    #region Messaging

    public void SendText(string text)
    {
        if (string.IsNullOrEmpty(text) || !IsConnected()) return;
        _ = SendTextAsync(text);
    }

    private async Task SendTextAsync(string text)
    {
        if (!IsConnected()) return;
        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            await ws.SendAsync(
                new ArraySegment<byte>(bytes),
                WebSocketMessageType.Text, true, CancellationToken.None
            );
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[WS] Send text failed: {e.Message}");
        }
    }

    public void SendMode(string mode)
    {
        var payload = JsonUtility.ToJson(new WsTextMessage { type = "mode", value = mode });
        SendText(payload);
    }

    public void SendSpeakText(string text)
    {
        if (string.IsNullOrEmpty(text)) return;
        StartCoroutine(SpeakSequence(text));
    }

    private IEnumerator SpeakSequence(string text)
    {
        SendMode("direct");
        yield return new WaitForSeconds(0.15f);
        SendText(text);
    }

    #endregion

    #region Microphone

    public void StartMicWithMode()
    {
        if (micOn) return;
        SendMode("llm");
        ToggleMic();
    }

    public void StopMic()
    {
        if (!micOn) return;
        ToggleMic();
    }

    public void ToggleMic()
    {
        _ = ToggleMicAsync();
    }

    private async Task ToggleMicAsync()
    {
        if (!micOn)
        {
            if (Microphone.devices.Length == 0)
            {
                Debug.LogError("[MIC] No microphone found");
                return;
            }

            micDevice = Microphone.devices[0];
            micClip = Microphone.Start(micDevice, true, 1, SAMPLE_RATE);

            int timeout = 0;
            while (Microphone.GetPosition(micDevice) <= 0 && timeout < 100)
            {
                await Task.Delay(10);
                timeout++;
            }

            if (timeout >= 100)
            {
                Debug.LogError("[MIC] Microphone failed to start");
                Microphone.End(micDevice);
                micClip = null;
                return;
            }

            micOn = true;
            lastSamplePosition = 0;
            micBuffer.Clear();
            UpdateMicUI(true);

            if (micCoroutine != null) StopCoroutine(micCoroutine);
            micCoroutine = StartCoroutine(MicStreamLoop());
        }
        else
        {
            if (micCoroutine != null)
            {
                StopCoroutine(micCoroutine);
                micCoroutine = null;
            }

            Microphone.End(micDevice);
            micClip = null;
            micOn = false;
            lastSamplePosition = 0;
            micBuffer.Clear();
            UpdateMicUI(false);
        }
    }

    private IEnumerator MicStreamLoop()
    {
        var wait = new WaitForSecondsRealtime(AUDIO_INTERVAL);
        while (micOn && micClip != null)
        {
            CaptureAndSendMicAudio();
            yield return wait;
        }
    }

    private void CaptureAndSendMicAudio()
    {
        if (micClip == null || !Microphone.IsRecording(micDevice)) return;

        int pos = Microphone.GetPosition(micDevice);
        if (pos == lastSamplePosition) return;

        int count = pos >= lastSamplePosition
            ? pos - lastSamplePosition
            : micClip.samples - lastSamplePosition + pos;

        if (count <= 0) return;

        var samples = new float[count];
        micClip.GetData(samples, lastSamplePosition);

        for (int i = 0; i < count; i++)
            micBuffer.Add(samples[i]);

        lastSamplePosition = pos;

        while (micBuffer.Count >= CHUNK_SIZE)
        {
            var chunk = new float[CHUNK_SIZE];
            micBuffer.CopyTo(0, chunk, 0, CHUNK_SIZE);
            micBuffer.RemoveRange(0, CHUNK_SIZE);
            _ = SendAudioChunk(chunk);
        }
    }

    private async Task SendAudioChunk(float[] samples)
    {
        if (!IsConnected()) return;

        byte[] pcm16 = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short v = (short)(Mathf.Clamp(samples[i], -1f, 1f) * 32767);
            pcm16[i * 2] = (byte)(v & 0xFF);
            pcm16[i * 2 + 1] = (byte)((v >> 8) & 0xFF);
        }

        byte[] message = new byte[1 + pcm16.Length];
        message[0] = 0x10;
        Buffer.BlockCopy(pcm16, 0, message, 1, pcm16.Length);

        try
        {
            await ws.SendAsync(
                new ArraySegment<byte>(message),
                WebSocketMessageType.Binary, true, CancellationToken.None
            );
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[WS] Audio send failed: {e.Message}");
        }
    }

    private void UpdateMicUI(bool enabled)
    {
        if (micBtnImage == null) return;
        var img = micBtnImage.GetComponent<Image>();
        if (img != null)
        {
            img.color = enabled
                ? new Color(0f, 1f, 0.0667f, 0.5f)
                : new Color(0.4157f, 0.4157f, 0.4157f, 0.5f);
        }
    }

    #endregion

    #region Cleanup

    async void OnDestroy()
    {
        if (micCoroutine != null)
        {
            StopCoroutine(micCoroutine);
            micCoroutine = null;
        }

        if (micOn && micDevice != null)
            Microphone.End(micDevice);

        cts?.Cancel();

        if (ws != null && ws.State == WebSocketState.Open)
        {
            try { await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None); }
            catch (Exception) { }
        }

        cts?.Dispose();
    }

    #endregion

    #region Serialization

    [Serializable]
    private class WsTextMessage
    {
        public string type;
        public string value;
        public string token;
        public string session_id;
    }

    #endregion
}
