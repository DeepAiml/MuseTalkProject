using System;
using System.Collections.Generic;
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
    [SerializeField] private float audioSpeedMultiplier = 1f;

    private const int AUDIO_SAMPLE_RATE = 22050;
    private const int MIN_BUFFER_SAMPLES = 3200;
    private const int STREAMING_BUFFER_SIZE = 88200;

    private Queue<byte[]> messageQueue = new Queue<byte[]>();
    private Texture2D videoTexture;

    private List<float> audioBuffer = new List<float>();
    private readonly object audioLock = new object();
    private AudioClip streamingClip;
    private bool isStreamingActive = false;
    private float audioReadPosition = 0f;

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
        InitializeAudio();
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

    #region Audio Setup

    private void InitializeAudio()
    {
        if (audioSource == null)
        {
            Debug.LogError("[Audio] AudioSource is not assigned");
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
            "StreamingAudio",
            STREAMING_BUFFER_SIZE,
            1,
            AUDIO_SAMPLE_RATE,
            true,
            OnAudioRead,
            OnAudioSetPosition
        );

        audioSource.clip = streamingClip;
    }

    #endregion

    #region Audio Streaming

    private void OnAudioRead(float[] data)
    {
        lock (audioLock)
        {
            int i = 0;
            while (i < data.Length)
            {
                int idx = Mathf.FloorToInt(audioReadPosition);
                if (idx < audioBuffer.Count)
                {
                    data[i] = audioBuffer[idx];
                    audioReadPosition += audioSpeedMultiplier;
                    i++;
                }
                else
                {
                    break;
                }
            }

            for (; i < data.Length; i++)
                data[i] = 0f;

            int consumed = Mathf.Min(Mathf.FloorToInt(audioReadPosition), audioBuffer.Count);
            if (consumed > 0)
            {
                audioBuffer.RemoveRange(0, consumed);
                audioReadPosition -= consumed;
            }

            if (audioBuffer.Count == 0)
                audioReadPosition = 0f;
        }
    }

    private void OnAudioSetPosition(int newPosition) { }

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
            audioSource.Stop();

        isStreamingActive = false;
    }

    #endregion

    #region Packet Processing

    public void OnBinaryMessage(byte[] data)
    {
        if (data == null || data.Length == 0) return;
        messageQueue.Enqueue(data);
    }

    private void HandlePacket(byte[] data)
    {
        if (data == null || data.Length < 1) return;

        switch (data[0])
        {
            case 0x01:
                HandleImagePacket(data);
                break;
            case 0x02:
                HandleAudioPacket(data);
                break;
            case 0x03:
            case 0x12:
                StopAllAudio();
                break;
        }
    }

    private void HandleImagePacket(byte[] data)
    {
        int jpegOffset = 5;
        if (data.Length <= jpegOffset) return;

        int jpegLength = data.Length - jpegOffset;
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
            Debug.LogError("[Video] JPEG decode failed");
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
    }

    #endregion

    void OnDestroy()
    {
        StopAllAudio();
    }
}
