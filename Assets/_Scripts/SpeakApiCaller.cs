using UnityEngine;
using System.Runtime.InteropServices;

public class SpeakData
{
    public string text;
    public string emotion;
    public int mic;
}

public class SpeakApiCaller : MonoBehaviour
{
    public MuseWsConnection _wsConnection;
    private string _authToken = "";

#if UNITY_WEBGL && !UNITY_EDITOR
    [System.Runtime.InteropServices.DllImport("__Internal")] private static extern void OnTokenExpired();
    [System.Runtime.InteropServices.DllImport("__Internal")] private static extern void OnSpeakRequestCompleted();
#endif

    void Awake()
    {

        //_wsConnection.OnBtn();
        gameObject.name = "SpeakApiCaller";
        DontDestroyOnLoad(gameObject);
        Debug.Log("SpeakApiCaller ready");
    }

    public void OnSpeakApiCall(string text, string emotion, string audioType)
    {
        Debug.Log("Speak api call from JS:");
        Debug.Log("Text: " + text);
        Debug.Log("Emotion: " + emotion);
        Debug.Log("Type: " + audioType);
        Speak(text, emotion, audioType);

    }
    void startmic()
    {
        //_wsConnection.StartMic();
    }

    public void OnSpeakApiCallWeb(string data)
    {
        Debug.Log("Raw data from JS: " + data);
        if (string.IsNullOrEmpty(data))
        {
            OnSpeakApiCall("", "neutral", "");
            return;
        }

        // Expected format: text|emotion
        string[] parts = data.Split('|');
        string text = parts.Length > 0 ? parts[0] : "";
        string emotion = parts.Length > 1 ? parts[1] : "neutral";
        string audioType = parts.Length > 2 ? parts[2] : "";
        OnSpeakApiCall(text, emotion, audioType);
        //  OnSpeakApiCall(text, emotion);
    }

    private void Speak(string text, string emotion, string audioType)
    {
        if (_wsConnection == null)
        {
            Debug.LogError("WsConnection not assigned!");
            return;
        }
        //_wsConnection.SetAuthToken(_authToken);
        _wsConnection.OnSpeakApiCallThroughWsConnection(text, emotion, audioType);
    }

    public void EnableMic()
    {
         //Microphone.Start(null, true, 10, 16000);

#if !UNITY_WEBGL || UNITY_EDITOR
        Microphone.Start(null, true, 10, 16000);
#endif


        Debug.Log("Enable mic from angular");
        //_wsConnection.micOn = false;
        _wsConnection.MakeDefaultSpeak("llm");
        _wsConnection.ToggleMic();
        //_wsConnection.micBtnImage.transform.localScale = Vector3.zero;
    }


    public void DisableMic()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        //Microphone.End(null);
#endif

         //Microphone.End(null);
        Debug.Log("disEnable mic from angular");
        //_wsConnection.micOn = true;
        _wsConnection.ToggleMic();
        //_wsConnection.micBtnImage.transform.localScale = Vector3.zero;

    }



    public void SetAuthToken(string combinedData)
    {
        // Split the combined data
        string[] parts = combinedData.Split('|');

        if (parts.Length == 2)
        {
            string token = parts[0];
            _wsConnection.token = "";
            _wsConnection.token = token;
            string session = parts[1];
            _wsConnection.session_Id = "";
            _wsConnection.session_Id = session;

            //_authToken = token;
            Debug.Log("Auth token received in Unity : " + token + "....." + session);
#if UNITY_WEBGL && !UNITY_EDITOR
Debug.Log("Auth token received in Unity : " + token + "....." + session);
#endif
            if (_wsConnection != null)
            {
                _wsConnection.UpdateToken(token, session, 1f);
            }
        }
    }

    // Call this when backend reports token invalid
    public void NotifyTokenExpired()
    {
        Debug.Log("Notified unity: token invalid/expired");
#if UNITY_WEBGL && !UNITY_EDITOR
        OnTokenExpired();   // Send event to browser
        Debug.Log("Notified browser: token invalid/expired");
#endif
    }

    public void TestTokenExpire(string dummy)
    {
        Debug.Log("Testing token expiry manually from Angular");
        Debug.Log(dummy);
        NotifyTokenExpired();
    }

    public void UpdateSpeakRequestCompleted()
    {
        Debug.Log("Notified unity: speak completed");
#if UNITY_WEBGL && !UNITY_EDITOR
        OnSpeakRequestCompleted();   // 🔥 this now works
        Debug.Log("Notified browser: speak completed");
#endif

    }

    
}

