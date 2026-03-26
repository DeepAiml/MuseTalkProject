using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.WebSockets;
using System.Runtime.InteropServices;
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

    [Header("UI")]
    public Button micBtnImage;
    
    [Header("State")]
    public bool micOn = false;
    public string token = "";
    public string session_Id = "";
    public string webgl_clientId = "-1";

#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")] private static extern void UnlockAudio();
    [DllImport("__Internal")] private static extern int InitializeAudioPlayback();
    [DllImport("__Internal")] private static extern int MuseWebSocketConnect(string url, string gameObjectName);
    [DllImport("__Internal")] private static extern int MuseWebSocketSendBinary(byte[] data, int length);
    [DllImport("__Internal")] private static extern int MuseWebSocketSendText(string text);
    [DllImport("__Internal")] private static extern void MuseWebSocketClose();
    [DllImport("__Internal")] private static extern int MuseWebSocketGetState();
    [DllImport("__Internal")] private static extern void MuseToggleMic(int micOn);
    [DllImport("__Internal")] private static extern int MuseIsMicOn();
    [DllImport("__Internal")] private static extern void DispatchSocketConnectedEvent();
    [DllImport("__Internal")] private static extern void DispatchSocketDisconnectedEvent(string message);
#else
    private ClientWebSocket ws;
    private AudioClip micClip;
    private int lastSamplePosition = 0;
    private List<float> audioBuffer = new List<float>();
    private float[] readBuffer;
    private CancellationTokenSource cts;
    private string micDevice;
    private Coroutine micCoroutine;
    private float[] wrapBuffer;
    private const float AUDIO_INTERVAL = 0.02f;
#endif

    async void Start()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        int result = MuseWebSocketConnect(WS_URL, gameObject.name);
        if (result == 1)
        {
            InitializeAudioPlayback();
            DispatchSocketConnectedEvent();
        }
        else
        {
            Debug.LogError("[WebGL] Failed to create WebSocket");
        }
#else
        cts = new CancellationTokenSource();
        ws = new ClientWebSocket();

        try
        {
            await ws.ConnectAsync(new Uri(WS_URL), CancellationToken.None);
            _ = ReceiveLoop();
        }
        catch (Exception e)
        {
            Debug.LogError($"[WS] Connection failed: {e.Message}");
        }
#endif
        UpdateMicUI(false);
    }

    #region Token Management

    public void UpdateToken(string token, string session_Id, float delay)
    {
        StartCoroutine(WaitForSocketAndSendToken(token, session_Id, delay));
    }

    private IEnumerator WaitForSocketAndSendToken(string token, string session_Id, float delay)
    {
        yield return new WaitForSeconds(delay);
        StartCoroutine(SendTokenCoroutine(token, session_Id));
    }

    private IEnumerator SendTokenCoroutine(string token, string session_id)
    {
        try
        {
            var tokenMessage = new TokenMessage
            {
                type = "submit_token",
                token = token,
                session_id = session_id
            };
            string jsonMessage = JsonUtility.ToJson(tokenMessage);

#if UNITY_WEBGL && !UNITY_EDITOR
            MuseWebSocketSendText(jsonMessage);
#else
            if (ws != null && ws.State == WebSocketState.Open)
            {
                byte[] messageBytes = Encoding.UTF8.GetBytes(jsonMessage);
                var sendTask = ws.SendAsync(
                    new ArraySegment<byte>(messageBytes),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );
                while (!sendTask.IsCompleted) { }
            }
#endif
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending token: {e.Message}");
        }
        yield return null;
    }

    [System.Serializable]
    public class TokenMessage
    {
        public string type;
        public string token;
        public string session_id;
    }

    #endregion

    #region Microphone

    public void ToggleMic()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        UnlockAudio();
        InitializeAudioPlayback();
        
        int jsMicOn = MuseIsMicOn();
        MuseToggleMic(jsMicOn == 0 ? 1 : 0);
#else
        _ = ToggleMicNative();
#endif
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

#if !UNITY_WEBGL || UNITY_EDITOR
    private async Task ToggleMicNative()
    {
        if (!micOn)
        {
            micDevice = Microphone.devices.Length > 0 ? Microphone.devices[0] : null;
            if (micDevice == null)
            {
                Debug.LogError("[MIC] No microphone found");
                return;
            }

            micClip = Microphone.Start(micDevice, true, 1, SAMPLE_RATE);

            int timeout = 0;
            while (Microphone.GetPosition(micDevice) <= 0 && timeout < 100)
            {
                await Task.Delay(10);
                timeout++;
            }

            if (timeout >= 100)
            {
                Debug.LogError("[MIC] Failed to start");
                Microphone.End(micDevice);
                micClip = null;
                return;
            }

            micOn = true;
            lastSamplePosition = 0;
            audioBuffer.Clear();
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
            audioBuffer.Clear();
            readBuffer = null;
            wrapBuffer = null;
            UpdateMicUI(false);
        }
    }

    private IEnumerator MicStreamLoop()
    {
        var wait = new WaitForSecondsRealtime(AUDIO_INTERVAL);
        while (micOn && micClip != null)
        {
            StreamMicAudio();
            yield return wait;
        }
    }

    private void StreamMicAudio()
    {
        if (micClip == null || !Microphone.IsRecording(micDevice)) return;

        int currentPosition = Microphone.GetPosition(micDevice);
        if (currentPosition < 0 || currentPosition > micClip.samples || currentPosition == lastSamplePosition)
            return;

        int sampleCount = currentPosition < lastSamplePosition
            ? micClip.samples - lastSamplePosition + currentPosition
            : currentPosition - lastSamplePosition;

        if (sampleCount <= 0) return;

        if (readBuffer == null || readBuffer.Length < sampleCount)
            readBuffer = new float[sampleCount];

        float[] samples = readBuffer;

        if (currentPosition < lastSamplePosition)
        {
            int firstPart = micClip.samples - lastSamplePosition;
            int total = firstPart + currentPosition;

            if (wrapBuffer == null || wrapBuffer.Length < total)
                wrapBuffer = new float[total];

            micClip.GetData(wrapBuffer, lastSamplePosition);
            micClip.GetData(wrapBuffer, 0);
            Array.Copy(wrapBuffer, 0, samples, 0, total);
        }
        else
        {
            micClip.GetData(samples, lastSamplePosition);
        }

        for (int i = 0; i < sampleCount; i++)
            audioBuffer.Add(samples[i]);

        while (audioBuffer.Count >= CHUNK_SIZE)
        {
            float[] chunk = new float[CHUNK_SIZE];
            audioBuffer.CopyTo(0, chunk, 0, CHUNK_SIZE);
            audioBuffer.RemoveRange(0, CHUNK_SIZE);
            _ = SendAudioBinary(chunk);
        }

        lastSamplePosition = currentPosition;
    }

    private async Task SendAudioBinary(float[] samples)
    {
        if (ws == null || ws.State != WebSocketState.Open) return;

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
            await ws.SendAsync(new ArraySegment<byte>(message),
                WebSocketMessageType.Binary, true, CancellationToken.None);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[WS] Send failed: {e.Message}");
        }
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
                        if (result.MessageType == WebSocketMessageType.Close) return;
                        ms.Write(buffer, 0, result.Count);
                    } while (!result.EndOfMessage);

                    if (result.MessageType == WebSocketMessageType.Binary)
                    {
                        byte[] data = ms.ToArray();
                        if (data.Length > 0 && ImageConvert.Instance != null)
                        {
                            ImageConvert.Instance.OnWebSocketMessage(data);
                        }
                    }
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception e)
        {
            Debug.LogError($"[WS] Error: {e}");
        }
    }

    async void OnDestroy()
    {
        if (micOn) Microphone.End(micDevice);
        cts?.Cancel();

        if (ws != null && ws.State == WebSocketState.Open)
        {
            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
        }

        if (micCoroutine != null)
        {
            StopCoroutine(micCoroutine);
            micCoroutine = null;
        }

        cts?.Dispose();
    }
#endif

    #endregion

    #region WebGL Callbacks

    public void OnWebSocketConnected(string message)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        InitializeAudioPlayback();
#endif
    }

    public void OnWebSocketClosed(string message)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        MuseToggleMic(0);
#endif
    }

    public void OnWebSocketError(string message)
    {
        Debug.LogError($"[WebGL] WebSocket error: {message}");
    }

    public void OnWebSocketBinary(string ptrAndLength)
    {
        ImageConvert.Instance?.OnWebSocketBinary(ptrAndLength);
    }

    public void OnMicrophoneStarted(string message)
    {
        micOn = true;
        UpdateMicUI(true);
    }

    public void OnMicrophoneStopped(string message)
    {
        micOn = false;
        UpdateMicUI(false);
    }

    public void OnMicrophoneError(string message)
    {
        Debug.LogError($"[WebGL] Microphone error: {message}");
        UpdateMicUI(false);
    }

    #endregion

    #region Speak API

    public void OnClickSpeakBtn()
    {
        StartCoroutine(MakeDefaultToDirectSpeak("direct"));
        StartCoroutine(WaitAndCallSpeakApi());
    }

    public void OnSpeakApiCallThroughWsConnection(string text, string emotion, string audioType)
    {
        MakeDefaultSpeak("direct");
        StartCoroutine(WaitAndCallSpeakApi());
    }

    public void MakeDefaultSpeak(string valueSubmit)
    {
        StartCoroutine(MakeDefaultToDirectSpeak(valueSubmit));
    }

    private IEnumerator WaitAndCallSpeakApi()
    {
        yield return new WaitForSeconds(0.1f);
        CallSpeakApi("Hello, I am from Dataviv Technology, and I am glad to meet you", webgl_clientId);
    }

    private IEnumerator MakeDefaultToDirectSpeak(string valueSubmit)
    {
        yield return new WaitForSeconds(0.1f);
        CallSpeakApiDefault("", webgl_clientId, valueSubmit);
    }

    public async Task CallSpeakApi(string text, string clientId)
    {
        text = "Hello, I am from Dataviv Technology, and I am glad to meet you";
        if (string.IsNullOrEmpty(text)) return;

        try
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            MuseWebSocketSendText(text);
#else
            byte[] bodyRaw = Encoding.UTF8.GetBytes(text);
            if (ws != null && ws.State == WebSocketState.Open)
            {
                await ws.SendAsync(
                    new ArraySegment<byte>(bodyRaw),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );
            }
#endif
        }
        catch (Exception ex)
        {
            Debug.LogError($"WebSocket send error: {ex.Message}");
        }
    }

    public async Task CallSpeakApiDefault(string text, string clientId, string valueSubmit)
    {
        text = "asda";
        if (string.IsNullOrEmpty(text)) return;

        var payload = new SpeakRequest { type = "mode", value = valueSubmit };
        string json = JsonUtility.ToJson(payload);

        try
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            MuseWebSocketSendText(json);
#else
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            if (ws != null && ws.State == WebSocketState.Open)
            {
                await ws.SendAsync(
                    new ArraySegment<byte>(bodyRaw),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );
            }
#endif
        }
        catch (Exception ex)
        {
            Debug.LogError($"WebSocket send error: {ex.Message}");
        }
    }

    [System.Serializable]
    private class SpeakRequest
    {
        public string type;
        public string value;
    }

    #endregion
}
