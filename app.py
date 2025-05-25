import os
import time
import pygame
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load environment variables ---
load_dotenv()

# --- Language Configuration ---
SUPPORTED_LANGUAGES = {
    "1": {"name": "English (US)", "code": "en-US", "gtts_lang": "en"},
    "2": {"name": "Hindi (India)", "code": "hi-IN", "gtts_lang": "hi"},
    "3": {"name": "Kannada (India)", "code": "kn-IN", "gtts_lang": "kn"},
    # Add more languages as needed, ensuring gTTS supports the base lang code
}

# --- Gemini API Key Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = None  # Initialize model as None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a generally available model. You can list models to find others.
        # For example: "gemini-1.5-flash-latest" or "gemini-1.0-pro" (if still available for you)
        # Refer to Google's documentation for the latest model names.
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        logger.info("Gemini AI model initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Gemini AI: {e}")
        logger.error("Please ensure your GEMINI_API_KEY is set correctly and is valid.")
        model = None # Ensure model is None if initialization fails
else:
    logger.warning("GEMINI_API_KEY environment variable not found.")
    logger.warning("Gemini AI features will be unavailable.")
    logger.warning("To enable, set it in your environment or a .env file: export GEMINI_API_KEY='YOUR_API_KEY'")


# --- Language Selector ---
def select_language():
    print("\nPlease select a language for speech recognition and TTS:")
    for key, lang_info in SUPPORTED_LANGUAGES.items():
        print(f"{key}: {lang_info['name']}")
    while True:
        choice = input("Enter the number of your choice: ")
        if choice in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[choice]
        else:
            print("Invalid choice. Please try again.")

# --- Speech Recognition ---
def recognize_speech_from_mic(recognizer, microphone, language_code="en-US", language_name="English"):
    if not isinstance(recognizer, sr.Recognizer):
        logger.error("Recognizer not an instance of sr.Recognizer")
        return {"success": False, "error": "Recognizer not properly initialized.", "transcription": None}
    if not isinstance(microphone, sr.Microphone):
        logger.error("Microphone not an instance of sr.Microphone")
        return {"success": False, "error": "Microphone not properly initialized.", "transcription": None}

    with microphone as source:
        print("\nAdjusting for ambient noise, please wait...")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            logger.error(f"Error adjusting for ambient noise: {e}")
            return {"success": False, "error": f"Ambient noise adjustment failed: {e}", "transcription": None}

        print(f"Listening in {language_name}. Please start speaking...")
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=20)
        except sr.WaitTimeoutError:
            logger.info("No speech detected within the time limit.")
            return {"success": True, "error": "No speech detected", "transcription": None}
        except Exception as e:
            logger.error(f"Error during listening: {e}")
            return {"success": False, "error": f"Listening failed: {e}", "transcription": None}

    print("Processing your speech...")
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio, language=language_code)
    except sr.RequestError as e:
        logger.error(f"API unavailable or request error: {e}")
        response["success"] = False
        response["error"] = "API unavailable. Check your internet connection or API quota."
    except sr.UnknownValueError:
        logger.info("Google Speech Recognition could not understand audio")
        response["error"] = "Unable to recognize speech."
    except Exception as e:
        logger.error(f"An unexpected error occurred during speech recognition: {e}")
        response["success"] = False
        response["error"] = f"Speech recognition failed: {e}"

    return response

# --- Text to Speech ---
def speak_text(text_to_speak, gtts_lang_code):
    if not text_to_speak or not isinstance(text_to_speak, str):
        logger.warning("No valid text provided to speak. Skipping TTS.")
        return

    temp_audio_file = "temp_speech.mp3"

    try:
        logger.info(f"Generating speech in {gtts_lang_code} for: '{text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}'")
        tts = gTTS(text=text_to_speak, lang=gtts_lang_code, slow=False)
        tts.save(temp_audio_file)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file)
        logger.info("Playing audio...")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10) # Keep the loop running smoothly
        # A short delay to ensure playback finishes before file operations
        time.sleep(0.5)

    except ValueError as e:
        if "is not a valid language code" in str(e):
            logger.error(f"gTTS does not support the language code '{gtts_lang_code}'. Error: {e}")
            print(f"Sorry, I cannot speak in {gtts_lang_code}. Please check the language support for gTTS.")
        else:
            logger.error(f"TTS/playback error: {e}")
            print(f"An error occurred during text-to-speech: {e}")
    except Exception as e:
        logger.error(f"TTS/playback error: {e}")
        print(f"An error occurred during text-to-speech: {e}")
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop() # Ensure music is stopped before quit
            pygame.mixer.quit()
        if os.path.exists(temp_audio_file):
            try:
                # Attempt to remove the file, adding a small delay if it fails initially
                for _ in range(3):
                    if not pygame.mixer.get_init() or not pygame.mixer.music.get_busy():
                        os.remove(temp_audio_file)
                        break
                    time.sleep(0.1)
                else:
                    if os.path.exists(temp_audio_file):
                        logger.warning(f"Could not delete temporary file: {temp_audio_file}. It might be in use.")
            except PermissionError:
                logger.warning(f"Permission denied to delete {temp_audio_file}.")
            except Exception as e_del:
                logger.error(f"Error deleting temp file {temp_audio_file}: {e_del}")

# --- Gemini AI Response ---
def get_ai_response(prompt):
    if not model:
        logger.warning("Gemini model is not initialized. Cannot get AI response.")
        return "Gemini AI model is not available. Please check your API key and configuration."
    if not prompt:
        logger.info("Received empty prompt for AI response.")
        return "I didn't receive any input. How can I help you?"

    try:
        logger.info(f"Sending to Gemini AI: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
        response = model.generate_content(prompt)

        # Check for blocked content
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason.name
            logger.warning(f"Gemini AI blocked the prompt. Reason: {reason}")
            return f"Sorry, your request was blocked by the AI for safety reasons ({reason}). Please try rephrasing."

        # Accessing text content
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif response.parts: # Fallback for structured responses
            all_text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
            if all_text_parts:
                return " ".join(all_text_parts).strip()

        logger.warning("Gemini returned a response, but it was empty or in an unexpected format.")
        # For debugging, you might want to log the full response:
        # logger.debug(f"Full Gemini Response: {response}")
        return "Sorry, I received an unusual response from Gemini AI."

    except Exception as e:
        logger.error(f"Error with Gemini API: {e}", exc_info=True)
        return "Sorry, I encountered an error while trying to get a response from Gemini AI."

# --- Main Program ---
if __name__ == "__main__":
    print("Welcome to the AI-Powered Multilingual Speech Assistant!")

    if not GEMINI_API_KEY:
        print("CRITICAL: GEMINI_API_KEY is not set. Gemini features will not work.")
        print("Please set the GEMINI_API_KEY environment variable or in a .env file.")
    elif not model:
        print("CRITICAL: Gemini AI model could not be initialized. Check API key and model name.")

    try:
        r = sr.Recognizer()
        mic = sr.Microphone()
        logger.info("Speech recognition components initialized.")
    except Exception as e:
        logger.error(f"Error initializing speech recognition components: {e}", exc_info=True)
        print("Fatal error: Could not initialize speech recognition. Please ensure you have a working microphone and necessary libraries (like PyAudio/PortAudio) installed.")
        exit()

    # Select language once at the beginning
    selected_lang_info = select_language()
    current_speech_lang_code = selected_lang_info["code"]
    current_speech_lang_name = selected_lang_info["name"]
    current_gtts_lang = selected_lang_info["gtts_lang"] # Use the specific gTTS lang code
    print(f"\nSelected language: {current_speech_lang_name} (Code: {current_speech_lang_code})")

    try:
        while True:
            user_choice = input(f"\nPress Enter to start speaking, 'c' to change language, or 'q' to quit: ").strip().lower()

            if user_choice == 'q':
                print("Exiting assistant. Goodbye!")
                break
            elif user_choice == 'c':
                selected_lang_info = select_language()
                current_speech_lang_code = selected_lang_info["code"]
                current_speech_lang_name = selected_lang_info["name"]
                current_gtts_lang = selected_lang_info["gtts_lang"]
                print(f"\nLanguage changed to: {current_speech_lang_name} (Code: {current_speech_lang_code})")
                continue
            # else (if Enter is pressed or any other key) -> proceed to listen

            speech_result = recognize_speech_from_mic(r, mic, current_speech_lang_code, current_speech_lang_name)

            if speech_result:
                recognized_text = speech_result["transcription"]
                if speech_result["success"] and recognized_text:
                    print("-" * 30)
                    print(f"You ({current_speech_lang_name}): {recognized_text}")
                    print("-" * 30)

                    if model: # Only proceed if Gemini model is available
                        ai_response = get_ai_response(recognized_text)
                        print("AI Response:")
                        print(ai_response)
                        speak_text(ai_response, current_gtts_lang)
                    else:
                        print("Gemini AI is not configured. Cannot get AI response.")
                        speak_text("I'm sorry, my AI brain is currently unavailable.", current_gtts_lang)

                elif speech_result["error"]:
                    print(f"Speech Recognition Info: {speech_result['error']}")
                    # Optionally, provide audio feedback for common non-critical errors
                    if speech_result["error"] == "Unable to recognize speech.":
                         speak_text("I didn't quite catch that. Could you please repeat?", current_gtts_lang)
                    elif speech_result["error"] == "No speech detected":
                         speak_text("I didn't hear anything. Please try speaking.", current_gtts_lang)

            else: # This case should ideally be covered by the error handling in recognize_speech_from_mic
                print("No speech was processed or an unexpected issue occurred with speech recognition.")

    except KeyboardInterrupt:
        print("\nExiting application...")
    finally:
        # Clean up Pygame mixer if it was initialized
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        print("Application closed.")