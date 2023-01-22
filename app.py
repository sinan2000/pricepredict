import tkinter as tk
import main

def get_input():
    input_text = text_box.get("1.0", "end-1c")
    main.predict(input_text)

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Text Input")

text_box = tk.Text(root, height=10, width=30)
text_box.pack()

button = tk.Button(root, text="Get Input", command=get_input)
button.pack()

exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.pack()

root.mainloop()
