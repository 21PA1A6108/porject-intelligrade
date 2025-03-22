from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import google.generativeai as genai
from typing import List
import os
from starlette.requests import Request
import uvicorn
import hashlib
from cachetools import LRUCache
import re

# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyDPTfc3gwLPFL86KljNc"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

evaluations_cache = LRUCache(maxsize=100)
def get_cache_key(faculty_answer, student_answer, marking_scheme):
    """Generate a unique key using hashing to avoid large dictionary keys."""
    key_data = f"{faculty_answer}-{student_answer}-{marking_scheme}"
    return hashlib.md5(key_data.encode()).hexdigest()

def extract_text(img_path):
    img = Image.open(img_path)
    response = model.generate_content(["read the image and give the text exactly", img])
    for i in response.text :
        if i != "*":
            ans = "".join(response.text)
    return ans

def send_message(prompt):
    var = model.generate_content(prompt)
    return var.text

def evaluate(faculty_answer, student_answer,marking_scheme):
    cache_key = get_cache_key(faculty_answer, student_answer, marking_scheme)

    if cache_key in evaluations_cache:
        return evaluations_cache[cache_key]
    
    scheme_description = {
        "scheme1": 100,
        "scheme2": 80,
        "scheme3": 50,
    }
    max_marks = scheme_description.get(marking_scheme, 100)
    prompt = (
        f"Compare the following two answers: \n\n"
        f"Faculty Answer: {faculty_answer}\n\n"
        f"Student Answer: {student_answer}\n\n"
        "Evaluate the student's answer using the following criteria:\n"
         "1. **Accuracy**: How well does the student's answer align with the correct facts, concepts, and information presented in the faculty's answer? Are there any discrepancies or errors?\n"
        "2. **Relevance**: Does the student's answer stay on-topic and directly answer the question, similar to the faculty answer? Are there any irrelevant or off-topic details?\n"
        "3. **Completeness**: Does the student's answer cover all the key points that are present in the faculty answer? Are any major points missing?\n"
        "4. **topic**: Does the student's answer is related to faculty answer topic? Student answer must not be related to any other topics other than faculty answer topic\n"
        "5. **Conciseness vs. Over-explanation**: Does the student answer the question in a concise and focused manner, without unnecessary elaboration or missing important points? How does this compare to the faculty's answer?\n"
        f"Grade the answer out of {max_marks} marks. "
        f"### Expected Response Format ###\n"
        f"Marks: {{marks_obtained}} obtained: {{marks_obtained}}/{max_marks}, Grade: {{grade_obtained}}\n\n"
        f"Explanation:\n{{detailed_feedback}}\n\n"
        "Provide the result exactly in this format, where the marks and grade appear first, followed by the explanation."
    )
    data = send_message(prompt)
    evaluations_cache[cache_key] = data
    return data



@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/result", response_class=HTMLResponse)
async def get_result(
        request: Request,
        images: List[UploadFile] = File(...),
        faculty_answer: str = Form(...),
        marking_scheme: str = Form(...)
    ):

    image_paths = []
    results = []
    
    for image in images:
        file_location = f"images/{image.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(image.file.read())
        image_paths.append(file_location)
        student_answer = extract_text(file_location)
        result = evaluate(faculty_answer, student_answer, marking_scheme)
        results.append(result)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "faculty_answer": faculty_answer,
        "image_paths": image_paths,
        "results": results,
        "zip": zip,  # Pass the zip function to the template context
        "enumerate": enumerate  # Pass the enumerate function to the template context
    }) 


std = extract_text(r"images\test-2.png")
std1= extract_text(r"images\test-1.jpg")
std2= extract_text(r"images\test-3.png")

import sklearn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample ground truth and predicted marks
# Example format: [(faculty_answer, student_answer, expected_marks, marking_scheme)]
ground_truth = [
    ('The JRE is a software package that provides the necessary libraries, files, and components to run Java applications on a device. It includes The JVM Core libraries like Java Standard Library Additional files required for running Java programs The JRE does not include tools for developing Java applications it only provides what is needed to execute Java code. So  it ideal for end-users who want to run Java applications but not to develop them.', std, 90, 'scheme1'),
    ('Skillspire is a learning platform developed by a team of four using HTML, CSS, and JavaScript, designed to simplify the process of learning a course from scratch. The platform features a homepage with navigation sections like Home, Courses, About, Contact, and Login/Signup. The Courses page lists all available courses, and each course has a dedicated page providing an introduction, learning outcomes, a roadmap, video tutorials, and reference books—all consolidated on a single webpage for easy access. My role as a frontend developer involved creating the responsive Courses page, organizing course materials, and ensuring a user-friendly experience. The platform aims to save students time by offering all the resources they need in one place, promoting efficient and focused learning.', std1, 80, 'scheme1'),
    ('A single linked list is a linear data structure where each element, called a node, contains two parts: the data and a reference to the next node in the sequence. It is dynamic, meaning the size can grow or shrink as needed, without fixed limits like arrays. The list starts with a special reference called the head, which points to the first node, and the last nodes next pointer is None, marking the end of the list. One of the key features is that it is unidirectional, meaning you can only traverse the list from the head to the tail. Memory is allocated as required, making it more space-efficient than arrays. Operations such as insertion and deletion can be performed efficiently, especially at the head or tail. However, traversal requires linear time since you must follow the next pointer from one node to the next.',std2,65,'scheme2')
    # Add more samples here
]

# Define a tolerance range for accuracy
tolerance = 10

# Map marking scheme to max marks
scheme_description = {
    "scheme1": 100,
    "scheme2": 80,
    "scheme3": 50,
}

# Generate predictions using your evaluate function
predictions = []
sum_res=0
expected_marks_list = []
predicted_marks_list = []
for faculty_answer, student_answer, expected_marks, marking_scheme in ground_truth:
    # Get the max marks based on the marking scheme
    max_marks = scheme_description.get(marking_scheme, 100)  # Default to 100 if scheme is not found
    predicted_data = evaluate(faculty_answer, student_answer, marking_scheme)  # using your existing evaluate function
    match = re.search(r"Marks:\s*(\d+)", predicted_data, re.IGNORECASE)

    if match:
        predicted_marks = int(match.group(1))  # Extract marks obtained
    else:
        predicted_marks = 0  # Default if extraction fails

    # print(f"Extracted -> Obtained Marks: {obtained_marks}")
    expected_marks_list.append(expected_marks)
    predicted_marks_list.append(predicted_marks)
    print(predicted_marks,expected_marks)
    if(abs(predicted_marks-expected_marks)<=tolerance):
        sum_res+=1


# Calculate accuracy as the percentage of predictions within the tolerance range
accuracy = sum_res / len(ground_truth)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(expected_marks_list, label="Expected Marks", marker='o', color='blue')
plt.plot(predicted_marks_list, label="Predicted Marks", marker='x', color='red')
plt.title("Comparison of Expected and Predicted Marks")
plt.xlabel("Sample Index")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.show()


