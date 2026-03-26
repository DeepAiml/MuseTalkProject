using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;

public class ImageConvert : MonoBehaviour
{
    public static ImageConvert Instance;

    [Header("References")]
    public RawImage OutPut_Image;
    public AudioSource audioSource;

    [Header("Audio Settings")]
    [Range(0.5f, 4.0f)]
    [SerializeField] public float audioSpeedMultiplier = 2f;

    private const int AUDIO_SAMPLE_RATE = 22050;
    private const int MIN_BUFFER_SAMPLES = 3200;
    private const int STREAMING_BUFFER_SIZE = 88200;

    private Queue<byte[]> messageQueue = new Queue<byte[]>();
    private Texture2D videoTexture;

    private List<float> audioBuffer = new List<float>();
    private object audioLock = new object();
    private AudioClip streamingClip;
    private bool isStreamingActive = false;
    private int samplesProvided = 0;
    private float audioReadPosition = 0f;
    private int totalAudioPackets = 0;
    private int totalAudioSamples = 0;

#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")] private static extern void UnlockAudio();
    [DllImport("__Internal")] private static extern int InitializeAudioPlayback();
#endif

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    void Start()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        InitializeWebGLAudio();
#else
        InitializeAudio();
#endif
    }

    void Update()
    {
        int maxPerFrame = 5;
        while (maxPerFrame-- > 0 && messageQueue.Count > 0)
        {
            HandlePacket(messageQueue.Dequeue());
        }

        if (!isStreamingActive && audioBuffer.Count >= MIN_BUFFER_SAMPLES)
        {
            StartStreaming();
        }
    }

    #region Audio Initialization

    private void InitializeAudio()
    {
        if (audioSource == null)
        {
            Debug.LogError("AudioSource is NULL");
            return;
        }

        audioSource.Stop();
        audioSource.clip = null;
        audioSource.playOnAwake = false;
        audioSource.loop = true;
        audioSource.volume = 1.0f;
        audioSource.spatialBlend = 0f;
        audioSource.priority = 0;

        streamingClip = AudioClip.Create(
            "StreamingAudioClip",
            STREAMING_BUFFER_SIZE,
            1,
            AUDIO_SAMPLE_RATE,
            true,
            OnAudioRead,
            OnAudioSetPosition
        );

        audioSource.clip = streamingClip;
        audioSource.loop = true;
        isStreamingActive = false;
    }

#if UNITY_WEBGL && !UNITY_EDITOR
    private void InitializeWebGLAudio()
    {
        if (audioSource != null)
        {
            audioSource.enabled = false;
        }
    }
#endif

    #endregion

    #region Audio Streaming

    private void OnAudioRead(float[] data)
    {
        lock (audioLock)
        {
            int needed = data.Length;
            int available = audioBuffer.Count;

            if (available >= needed)
            {
                for (int i = 0; i < needed; i++)
                {
                    int sourceIndex = Mathf.FloorToInt(audioReadPosition);
                    data[i] = sourceIndex < audioBuffer.Count ? audioBuffer[sourceIndex] : 0f;
                    audioReadPosition += audioSpeedMultiplier;
                }

                int samplesConsumed = Mathf.FloorToInt(audioReadPosition);
                if (samplesConsumed > 0 && samplesConsumed <= audioBuffer.Count)
                {
                    audioBuffer.RemoveRange(0, samplesConsumed);
                    audioReadPosition -= samplesConsumed;
                }
                samplesProvided += needed;
            }
            else if (available > 0)
            {
                int outputIndex = 0;
                audioReadPosition = 0f;

                for (int i = 0; i < needed && outputIndex < needed; i++)
                {
                    int sourceIndex = Mathf.FloorToInt(audioReadPosition);
                    if (sourceIndex < audioBuffer.Count)
                    {
                        data[outputIndex++] = audioBuffer[sourceIndex];
                        audioReadPosition += audioSpeedMultiplier;
                    }
                    else break;
                }

                for (int i = outputIndex; i < needed; i++)
                {
                    data[i] = 0f;
                }

                audioBuffer.Clear();
                audioReadPosition = 0f;
                samplesProvided += outputIndex;
            }
            else
            {
                for (int i = 0; i < needed; i++)
                {
                    data[i] = 0f;
                }
                audioReadPosition = 0f;
            }
        }
    }

    private void OnAudioSetPosition(int newPosition)
    {
        samplesProvided = newPosition;
    }

    private void StartStreaming()
    {
        if (isStreamingActive || audioSource == null || streamingClip == null) return;
        if (audioBuffer.Count < MIN_BUFFER_SAMPLES) return;

        audioSource.Play();
        isStreamingActive = true;
    }

    private void StopAllAudio()
    {
        lock (audioLock)
        {
            audioBuffer.Clear();
            audioReadPosition = 0f;
        }

        if (audioSource != null && audioSource.isPlaying)
        {
            audioSource.Stop();
        }

        isStreamingActive = false;
        samplesProvided = 0;
    }

    private void EnsureAudioUnlocked()
    {
        if (audioSource == null) return;

#if UNITY_WEBGL && !UNITY_EDITOR
        try { UnlockAudio(); }
        catch (Exception) { }
#endif
    }

    #endregion

    #region WebSocket Message Handling

    public void OnWebSocketBinary(string msg)
    {
        var split = msg.Split(',');
        int ptr = int.Parse(split[0]);
        int len = int.Parse(split[1]);

        byte[] data = new byte[len];
        Marshal.Copy((IntPtr)ptr, data, 0, len);

        messageQueue.Enqueue(data);
    }

    public void OnWebSocketMessage(byte[] data)
    {
        if (data == null || data.Length == 0) return;
        messageQueue.Enqueue(data);
    }

    #endregion

    #region Packet Processing

    private void HandlePacket(byte[] data)
    {
        if (data == null || data.Length < 1) return;

        byte type = data[0];

        switch (type)
        {
            case 0x01:
                HandleImagePacket(data);
                break;
            case 0x02:
                HandleAudioPacket(data);
                break;
            case 0x03:
                StopAllAudio();
                EnsureAudioUnlocked();
                break;
            case 0x12:
                StopAllAudio();
                break;
        }
    }

    private void HandleImagePacket(byte[] data)
    {
        if (data.Length < 6) return;

        int jpegOffset = 5;
        int jpegLength = data.Length - jpegOffset;
        if (jpegLength <= 0) return;

        byte[] jpeg = new byte[jpegLength];
        Buffer.BlockCopy(data, jpegOffset, jpeg, 0, jpegLength);

        DisplayFrame(jpeg);
    }

    private void DisplayFrame(byte[] jpegBytes)
    {
        if (videoTexture == null)
        {
            videoTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false, false);
            videoTexture.wrapMode = TextureWrapMode.Clamp;
            videoTexture.filterMode = FilterMode.Bilinear;

            OutPut_Image.texture = videoTexture;
            OutPut_Image.material = null;
            OutPut_Image.color = Color.white;
            OutPut_Image.uvRect = new Rect(0, 0, 1, 1);
        }

        if (!videoTexture.LoadImage(jpegBytes, false))
        {
            Debug.LogError("JPEG decode failed");
            return;
        }

        videoTexture.Apply(false, false);
        OutPut_Image.SetAllDirty();
    }

    private void HandleAudioPacket(byte[] data)
    {
        int pcmOffset = 1;
        int pcmBytes = data.Length - pcmOffset;
        if (pcmBytes <= 0) return;

#if UNITY_WEBGL && !UNITY_EDITOR
        totalAudioPackets++;
        return;
#else
        int sampleCount = pcmBytes / 2;
        float[] samples = new float[sampleCount];

        int idx = pcmOffset;
        for (int i = 0; i < sampleCount; i++)
        {
            short s = (short)(data[idx] | (data[idx + 1] << 8));
            samples[i] = s / 32768f;
            idx += 2;
        }

        lock (audioLock)
        {
            audioBuffer.AddRange(samples);
        }

        totalAudioPackets++;
        totalAudioSamples += samples.Length;

        if (totalAudioPackets == 1)
        {
            EnsureAudioUnlocked();
            StartStreaming();
        }
#endif
    }

    #endregion

    void OnDestroy()
    {
        StopAllAudio();
    }
}
