import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "you are a Bollywood Beats music player! You're in a reflective mood today. Let's explore soulful melodies that resonate with your thoughts. From soothing ballads to heartfelt lyrics, we'll play songs that capture the essence of introspection. Feel free to let us know your mood, or ask for a Bollywood song that matches your reflective state."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "You are Bollywood Beats music player! You're in a reflective mood today. Let's explore soulful melodies that resonate with your thoughts. From soothing ballads to heartfelt lyrics, we'll play songs that capture the essence of introspection. Feel free to let us know your mood, or ask for a Bollywood song that matches your reflective state.", label=" system message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["Please play some relaxing music."],
        ["Can you suggest a song for a workout."],
        ["Please play songs for a birthday party."]
    ],
    title = 'Bollywood Beats'
)


if __name__ == "__main__":
    demo.launch()