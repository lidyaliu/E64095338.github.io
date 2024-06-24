from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Function to compress image using SVD
def compress_image(image_path, k):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image)
    
    # Perform SVD
    U, S, V = np.linalg.svd(image_matrix, full_matrices=False)
    
    # Truncate matrices based on k
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]
    
    # Reconstruct compressed image matrix
    compressed_image_matrix = np.dot(U_k, np.dot(S_k, V_k))
    
    # Convert compressed matrix back to image
    compressed_image = Image.fromarray(compressed_image_matrix.astype('uint8'), 'L')
    
    # Save compressed image to byte stream
    output = io.BytesIO()
    compressed_image.save(output, format='PNG')
    output.seek(0)
    
    return output

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html', compressed_image=None)

# Route to handle file upload and compression
@app.route('/compress', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file and k value from the form
        file = request.files['file']
        k = int(request.form['k'])
        
        if file:
            # Save the uploaded file temporarily
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            
            # Perform image compression
            compressed_image = compress_image(image_path, k)
            
            # Delete the uploaded file after compression
            os.remove(image_path)
            
            # Send compressed image as a downloadable file
            return send_file(compressed_image, mimetype='image/png', as_attachment=True, attachment_filename='compressed_image.png')
    
    # Render index.html if something goes wrong
    return render_template('index.html', compressed_image=None)

if __name__ == '__main__':
    app.run(debug=True)
