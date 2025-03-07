import tkinter as tk
from PIL import Image, ImageTk

class PixelCoordinateApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Pixel Coordinate Viewer")

        # Load the image
        self.image = Image.open(image_path)
        
        # Resize the image to fit within the screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        image_width, image_height = self.image.size
        
        # Calculate the ratio to resize the image
        ratio = 1#min(screen_width / image_width, screen_height / image_height)
        new_width = int(image_width * ratio)
        new_height = int(image_height * ratio)
        
        # Resize the image using LANCZOS resampling
        self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert the image to a format Tkinter can use
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Create a canvas to display the image
        self.canvas = tk.Canvas(root, width=new_width, height=new_height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Bind mouse move event
        self.canvas.bind("<Motion>", self.show_coordinates)  # Use "<Motion>" to track mouse movement without clicking

        # Create a label to display coordinates
        self.coord_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.coord_label.pack()

    def show_coordinates(self, event):
        # Get the mouse position (this is already in canvas coordinates)
        x, y = event.x, event.y
        # Display the coordinates
        self.coord_label.config(text=f"Pixel Coordinates: ({x}, {y})")

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelCoordinateApp(root, "frame_0147.jpg")  # Replace with your image path
    root.mainloop()