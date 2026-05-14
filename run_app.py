from pyngrok import ngrok
import subprocess

ngrok.set_auth_token("3DJCFOPy3VWWgxEPcJc8vaJKgfR_3uSbsKLNg6SRgAnbuDMqv")

ngrok.kill()

subprocess.Popen(["streamlit", "run", "Healthcare-Disease-Prediction/app.py"])

public_url = ngrok.connect(8501)
print(f"Streamlit App URL: {public_url}")