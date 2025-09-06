import subprocess
import sys


repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

def main():
    """
    Main function to launch the AI Study Assistant Streamlit application.

    This script acts as a simple, user-friendly wrapper to start the app.
    It uses the subprocess module to execute the `streamlit run` command,
    targeting the main application script in the `src/app/` directory.
    """
    print("=====================================================")
    print("🚀 Launching the AI Study Assistant...")
    print("=====================================================")


    command = [sys.executable, "-m", "streamlit", "run", "src/app/main.py"]

    try:

        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\n❌ ERROR: 'streamlit' command not found.")
        print("Please ensure Streamlit is installed in your environment (`pip install streamlit`).")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: The Streamlit application exited with an error: {e}")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")

if __name__ == "__main__":
    main()
