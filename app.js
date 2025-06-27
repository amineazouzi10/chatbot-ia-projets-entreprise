const recordButton = document.getElementById('recordButton');
const sendTextButton = document.getElementById('sendTextButton');
const textInput = document.getElementById('textInput');
const statusDiv = document.getElementById('status');
const chatLog = document.getElementById('chatLog');

// Loader as chat bubble
function appendBotLoader() {
    removeBotLoader();
    const loaderWrapper = document.createElement('div');
    loaderWrapper.classList.add('message', 'bot-message');
    loaderWrapper.setAttribute('id', 'botLoaderMsg');

    const loaderDiv = document.createElement('div');
    loaderDiv.className = 'bot-loader';
    loaderDiv.innerHTML = `
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="generating-text">Génération de la réponse...</span>
    `;

    loaderWrapper.appendChild(loaderDiv);
    chatLog.appendChild(loaderWrapper);
    chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: 'smooth' });
}

function removeBotLoader() {
    const oldLoader = document.getElementById('botLoaderMsg');
    if (oldLoader) oldLoader.remove();
}

// Icons
const micIcon = '<i class="fas fa-microphone"></i>';
const stopIcon = '<i class="fas fa-stop"></i>';
const playIcon = '<i class="fas fa-play"></i>';
const playingIcon = '<i class="fas fa-pause"></i>';

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let currentSenderId = `user_${Math.random().toString(36).substr(2, 9)}`;

const BACKEND_URL = 'http://localhost:5001/process_input';

// Function to format text with line breaks and preserve formatting
function formatTextForDisplay(text) {
    // Convert line breaks to HTML line breaks and preserve spacing
    return text.replace(/\n/g, '<br>').replace(/  +/g, match => '&nbsp;'.repeat(match.length));
}

// Function to add messages to the chat log
function appendMessage(text, sender, audioUrl = null) {
    const messageWrapper = document.createElement('div');
    messageWrapper.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');

    const messageTextDiv = document.createElement('div');
    messageTextDiv.classList.add('message-text');

    // Use innerHTML instead of textContent to preserve formatting
    if (sender === 'bot') {
        messageTextDiv.innerHTML = formatTextForDisplay(text);
    } else {
        messageTextDiv.textContent = text; // Keep user messages as plain text for security
    }

    messageWrapper.appendChild(messageTextDiv);

    if (sender === 'bot' && audioUrl) {
        const playButton = document.createElement('button');
        playButton.classList.add('play-tts-button');
        playButton.innerHTML = playIcon;
        playButton.dataset.audioUrl = audioUrl;
        playButton.setAttribute('title', 'Lire la réponse');
        playButton.onclick = function() {
            const urlToPlay = this.dataset.audioUrl;
            if (urlToPlay) {
                const absoluteAudioUrl = urlToPlay.startsWith('/') ?
                    `${new URL(BACKEND_URL).origin}${urlToPlay}` : urlToPlay;
                const audio = new Audio(absoluteAudioUrl);
                const originalStatus = statusDiv.textContent;
                statusDiv.textContent = 'Lecture en cours...';
                this.innerHTML = playingIcon;
                this.disabled = true;

                audio.play();
                audio.onended = () => {
                    statusDiv.textContent = 'Prêt. Tapez ou utilisez le micro.';
                    this.innerHTML = playIcon;
                    this.disabled = false;
                };
                audio.onerror = (e) => {
                    console.error("Error playing audio:", e);
                    statusDiv.textContent = 'Erreur de lecture audio.';
                    this.innerHTML = playIcon;
                    this.disabled = false;
                };
            }
        };
        messageWrapper.appendChild(playButton);
    }

    chatLog.appendChild(messageWrapper);
    chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: 'smooth' });
}

// Function to display multiple bot messages with a small delay between them
function displayBotMessages(messages, delay = 1000) {
    removeBotLoader();

    messages.forEach((message, index) => {
        setTimeout(() => {
            appendMessage(message.text, 'bot', message.audio_url);
        }, index * delay);
    });
}

function showLoading(isLoading) {
    if (isLoading) {
        statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Traitement en cours...';
    } else {
        statusDiv.textContent = 'Prêt. Tapez ou utilisez le micro.';
    }
}

// Send text message (with loader bubble)
sendTextButton.onclick = () => {
    const message = textInput.value.trim();
    if (message) {
        appendMessage(message, 'user');
        appendBotLoader(); // Loader bubble just after user message
        sendMessageToBackend(message, null, false);
        textInput.value = '';
    }
};

textInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendTextButton.click();
    }
});

async function sendMessageToBackend(textMessage, audioBlob, isVoiceInput = false) {
    const formData = new FormData();
    formData.append('sender', currentSenderId);

    if (audioBlob) {
        formData.append('audio_data', audioBlob, 'user_audio.webm');
    } else if (textMessage) {
        formData.append('message', textMessage);
    }

    showLoading(true);

    try {
        const response = await fetch(BACKEND_URL, {
            method: 'POST',
            body: formData
        });

        showLoading(false);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: "Réponse non JSON du serveur" }));
            throw new Error(errorData.error || `Erreur serveur: ${response.status}`);
        }

        const data = await response.json();

        if (isVoiceInput && data.user_input_text) {
            appendMessage(data.user_input_text, 'user');
        }

        // Handle multiple messages from the backend
        if (data.messages && Array.isArray(data.messages)) {
            displayBotMessages(data.messages);
        } else {
            // Fallback for old format compatibility
            removeBotLoader();
            const botText = data.bot_reply_text || "Désolé, je n'ai pas de réponse.";
            appendMessage(botText, 'bot', data.audio_url);
        }

        if (data.error_tts) {
            console.warn("TTS Error from backend:", data.error_tts);
        }

    } catch (error) {
        removeBotLoader();
        showLoading(false);
        appendMessage(`Désolé, une erreur de communication est survenue: ${error.message}`, 'bot');
    }
}

// Voice recording logic
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    recordButton.onclick = async () => {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    appendBotLoader(); // Loader bubble for voice input too
                    const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                    sendMessageToBackend(null, audioBlob, true);
                };

                mediaRecorder.start();
                isRecording = true;
                recordButton.innerHTML = stopIcon;
                recordButton.classList.add('recording');
                recordButton.title = "Arrêter l'enregistrement";
                statusDiv.textContent = 'Enregistrement en cours...';
            } catch (err) {
                console.error('Error accessing microphone:', err);
                statusDiv.textContent = 'Erreur d\'accès au microphone: ' + err.message;
            }
        } else {
            mediaRecorder.stop();
            isRecording = false;
            recordButton.innerHTML = micIcon;
            recordButton.classList.remove('recording');
            recordButton.title = "Parler";
        }
    };
    recordButton.innerHTML = micIcon;
} else {
    statusDiv.textContent = "Désolé, votre navigateur ne supporte pas l'enregistrement audio.";
    recordButton.disabled = true;
    recordButton.innerHTML = micIcon;
}

statusDiv.textContent = 'Prêt. Tapez ou utilisez le micro.';
sendTextButton.innerHTML = '<i class="fas fa-paper-plane"></i>';