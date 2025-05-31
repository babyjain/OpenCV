# import pytesseract
# import cv2
# from pytesseract import Output
# import matplotlib.pyplot as plt
# img_path= "D:\INTERNSHIP\DEEP LEARNING\opencv\data\made_my.jpg"
# img=cv2.imread(img_path)
# if img is None:
#     print(f"no image is found:{img_path}")
#     exit()
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# _,thresh=cv2.threshold(gray,100,250,cv2.THRESH_BINARY)
# cv2.imwrite("processed_image.jpg",thresh)
# plt.imshow(thresh,cmap='gray')
# plt.axis("off")
# # plt.show()

# custom_config = r'--oem 3 --psm 6'
# text = pytesseract.image_to_string(thresh,config=custom_config)
# data = pytesseract.image_to_data(thresh,output_type=Output.DICT,config=custom_config)

# number_words = len(data['text'])
# if number_words == 0:
#     print('no_words_detected')
# else:
#     for i in range(number_words):
#         if int(data['conf'][i])>20:
#             x,y,w,h = data['left'][i],data['top'][i],data['width'][i],data['height'][i]
#             cv2.rectangle(img,(x,y),(x+w,y+h),color=(255,0,0),thickness=2)
#             print(f"word:{data['text'][i]}, confidence:{data['conf'][i]}")
#     img_rgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.axis("off")
#     plt.show()

# import cv2
# import matplotlib.pyplot as plt
# import easyocr

# # Load the image
# img_path = 'D:\INTERNSHIP\DEEP LEARNING\opencv\data\invoice.jpeg'
# img = cv2.imread(img_path)
# if img is None:
#     print(f"Image Not Found: {img_path}")
#     exit()

# # Convert image to grayscale and threshold
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 100, 250, cv2.THRESH_BINARY)

# # Save and show processed image
# cv2.imwrite("processed_image.jpg", thresh)
# plt.imshow(thresh, cmap='gray')
# plt.axis("off")
# # plt.show()

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Run OCR
# results = reader.readtext(thresh)

# # Check and display results
# if not results:
#     print('no_words_detected')
# else:
#     for (bbox, text, confidence) in results:
#         if confidence > 0.25:  # confidence ranges from 0 to 1
#             print(f"word: {text}, confidence: {confidence:.2f}")
#             # Draw rectangle
#             (top_left, top_right, bottom_right, bottom_left) = bbox
#             top_left = tuple(map(int, top_left))
#             bottom_right = tuple(map(int, bottom_right))
#             cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

#     # Show the final image with boxes
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.axis("off")
#     plt.show()
# import cv2
# import matplotlib.pyplot as plt
# import easyocr
# import re

# # Load image
# img_path = r'D:\INTERNSHIP\DEEP LEARNING\opencv\data\invoice.jpeg'
# img = cv2.imread(img_path)
# if img is None:
#     print(f"Image Not Found: {img_path}")
#     exit()

# # Preprocess
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# # OCR
# reader = easyocr.Reader(['en'], gpu=False)
# results = reader.readtext(thresh)

# # Regex patterns
# email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
# date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
# # Combine all text for global search
# full_text = " ".join([t[1] for t in results])
# clean_text = full_text.replace(" ", "").replace("\n", "").strip()

# # Match emails and dates
# emails = re.findall(email_pattern, clean_text)
# dates = re.findall(date_pattern, full_text, flags=re.IGNORECASE)

# found_any = False

# # Print matches
# for email in emails:
#     print(f"📧 Email: {email}")
#     found_any = True
# for date in dates:
#     print(f"📅 Date: {date}")
#     found_any = True

# # Draw boxes for matches
# for (bbox, text, conf) in results:
#     if conf > 0.25:
#         for email in emails:
#             if email in text.replace(" ", ""):
#                 top_left = tuple(map(int, bbox[0]))
#                 bottom_right = tuple(map(int, bbox[2]))
#                 cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
#         for date in dates:
#             if date in text:
#                 top_left = tuple(map(int, bbox[0]))
#                 bottom_right = tuple(map(int, bbox[2]))
#                 cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

# if not found_any:
#     print("❌ No email or date detected.")

# # Show result
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.axis("off")
# plt.show()
# import easyocr
# import cv2
# import re
# import matplotlib.pyplot as plt

# # Load the image
# image_path = r'data/invoice.jpeg'
# img = cv2.imread(image_path)

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply binary thresholding
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Initialize EasyOCR
# reader = easyocr.Reader(['en'], gpu=False)

# # Use original image (skip thresholding)
# results = reader.readtext(thresh)

# # Print detected text for debugging
# text_data = []
# for (bbox, text, confidence) in results:
#     print(f"[{confidence:.2f}] {text}")  # Debug
#     if confidence > 0.2:
#         text_data.append(text)

# full_text = ' '.join(text_data)

# # Relaxed regex for email
# # email_regex = r'\b[\w\.-]+\s*@\s*[\w\.-]+\s*\.\s*[a-zA-Z]{2,}\b'
# # email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
# relaxed_email_regex = r'\b[\w\.-]+\s*@\s*[\w\.-]+\s*[.\s]\s*[a-zA-Z]{2,}\b'
# emails = re.findall(relaxed_email_regex, full_text)
# emails = [email.replace(" ", "").replace("..", ".") for email in emails]


# # Extract dates
# date_regex = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
# dates = re.findall(date_regex, full_text)

# # Output results
# print("📅 Dates found:", dates)
# print("📧 Emails found:", emails)

# # Convert BGR to RGB for display with matplotlib
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Show the image with bounding boxes and text
# plt.figure(figsize=(12, 8))
# plt.imshow(img_rgb)
# plt.axis("off")
# plt.title("Detected Text with Bounding Boxes")
# plt.show()


import easyocr
import cv2
import re
import matplotlib.pyplot as plt

# Load the image
image_path = r'data/invoice.jpeg'  # <-- Make sure this path is correct
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold (optional)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# OCR on thresholded image
results = reader.readtext(thresh)

# Define relaxed regex patterns
email_pattern = r'\b[\w\.-]+\s*@\s*[\w\.-]+\s*[.\s]\s*[a-zA-Z]{2,}\b'
date_pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'

# Initialize result holders
emails_found = []
dates_found = []

# Scan OCR results
for (bbox, text, confidence) in results:
    if confidence < 0.2:
        continue

    # Clean text
    clean_text = text.strip()
    full_text = clean_text.replace(" ", "").replace("..", ".")

    # Match email
    if re.search(email_pattern, clean_text):
        email_match = re.search(email_pattern, clean_text)
        emails_found.append(email_match.group().replace(" ", "").replace("..", "."))

        # Draw green box for email
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(img, 'Email', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Match date
    if re.search(date_pattern, clean_text):
        date_match = re.search(date_pattern, clean_text)
        dates_found.append(date_match.group())

        # Draw blue box for date
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(img, 'Date', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display found emails and dates
print("📧 Emails found:", emails_found)
print("📅 Dates found:", dates_found)

# Display image with boxes
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Detected Emails and Dates")
plt.show()
