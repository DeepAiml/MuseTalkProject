using UnityEngine;

public class SpeakApiCaller : MonoBehaviour
{
    public MuseWsConnection _wsConnection;

    void Awake()
    {
        gameObject.name = "SpeakApiCaller";
        DontDestroyOnLoad(gameObject);
    }

    public void OnSpeakApiCallWeb(string data)
    {
        if (string.IsNullOrEmpty(data)) return;

        string[] parts = data.Split('|');
        string text = parts.Length > 0 ? parts[0] : "";
        string emotion = parts.Length > 1 ? parts[1] : "neutral";
        string audioType = parts.Length > 2 ? parts[2] : "";

        if (_wsConnection != null && !string.IsNullOrEmpty(text))
        {
            _wsConnection.SendSpeakText(text);
        }
    }

    public void EnableMic()
    {
        if (_wsConnection == null) return;
        _wsConnection.StartMicWithMode();
    }

    public void DisableMic()
    {
        if (_wsConnection == null) return;
        _wsConnection.StopMic();
    }

    public void SetAuthToken(string combinedData)
    {
        if (_wsConnection == null || string.IsNullOrEmpty(combinedData)) return;

        string[] parts = combinedData.Split('|');
        if (parts.Length >= 2)
        {
            string tokenValue = parts[0];
            string sessionValue = parts[1];
            _wsConnection.token = tokenValue;
            _wsConnection.session_Id = sessionValue;
            _wsConnection.UpdateToken(tokenValue, sessionValue, 1f);
        }
    }

    public void NotifyTokenExpired()
    {
        Debug.Log("[Auth] Token expired");
    }

    public void TestTokenExpire(string dummy)
    {
        NotifyTokenExpired();
    }

    public void UpdateSpeakRequestCompleted()
    {
        Debug.Log("[Speak] Request completed");
    }
}
