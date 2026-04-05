import React, { useState } from 'react';
import axios from 'axios';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import "./App.css";

function App() {
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [loading, setLoading] = useState(false); 

  const sessionId = "session1";  // Static session ID for now
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = SpeechRecognition ? new SpeechRecognition() : null;

  // Start recording when user clicks a button
  const startRecording = () => {
    if (recognition) {
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setQuestion(transcript);  // Set the recognized speech as the question
      };

      recognition.onend = () => {
        setIsListening(false);
        handleSend(); // Automatically send the question after recording ends
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognition.start();  // Start recording when button is clicked
    } else {
      alert("Speech recognition not supported in this browser.");
    }
  };

  // Function to send the question to the bot
  const handleSend = async () => {
    if (question.trim()) { 
      setLoading(true)
      console.log({ question, session_id: sessionId });  // Debugging log

      try {
        const res = await axios.post("http://localhost:8000/ask", { 
          question: question, 
          session_id: sessionId 
        });

        const botResponse = res.data.response;

        setConversation(prevConversation => {
          const newConversation = [...prevConversation, { question, botResponse }];
          return newConversation;
        });

        handlePlayAudio(botResponse);  // Play the bot's response

        setQuestion('');  // Clear input field

      } catch (error) {
        console.error('Error fetching response:', error);
      } finally {
        setLoading(false)
      }
    }
  };

  // Function to handle browser TTS for each bot response
  const handlePlayAudio = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 1;
    utterance.pitch = 1;
    speechSynthesis.speak(utterance);
  };

  return (
    <div className="App">
      <h1>CHATBOT</h1>
      <div className="chat-box">
        {conversation.map((msg, index) => (
          <div key={index} className="message-container">
            <div className="user-message">
              <p><strong>User: </strong> {msg.question}</p>
            </div>
            <div className="bot-message">
              <p><strong>Bot: </strong><Markdown remarkPlugins={[[remarkGfm, {singleTilde: true}]]}>
                {msg.botResponse}
              </Markdown>
              </p>
            </div>
          </div>
        ))}
      </div>
      <div>
        {loading && <p>Wait a second dude...</p>}
      </div>
      <div class="wrap">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask me something...😁"
      />
      <button className="btn" onClick={handleSend}> Send▶️ </button>
      </div>
      {/* Start recording when user clicks this button */}
      <button className="btn" onClick={startRecording} disabled={isListening}>
        {isListening ? 'Listening...' : '🎤 Start Recording'}
      </button>
      <p>{isListening ? 'Listening...' : 'Click to speak'}</p>
    </div>
  );
}

export default App;
