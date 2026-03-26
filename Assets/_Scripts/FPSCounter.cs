using UnityEngine;
using UnityEngine.UI;

public class FPSCounter : MonoBehaviour
{
    public Text fpsText;
    float smoothDelta;

    void Awake()
    {
        QualitySettings.vSyncCount = 0;  // Disable VSync
        Application.targetFrameRate = 20;
    }

    void Update()
    {
        smoothDelta = Mathf.Lerp(smoothDelta, Time.unscaledDeltaTime, 0.1f);
        if (Time.frameCount % 10 == 0)
        {
            fpsText.text = "FPS: " + Mathf.RoundToInt(1f / smoothDelta);
        }
    }
}