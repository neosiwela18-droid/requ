import cv2 as cv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient


man = InferenceHTTPClient(
                            api_url="API_URL",
                            api_key="API_KEY"
).configure(InferenceConfiguration(
                                    api_key_transport='header'
                                    ))

results = man.run_workflow(
                            workspace_name='YOUR_WORKSPACE',
                            workflow_id="YOUR_WORKFLOW_ID",
                            images = {'image': r'pics\20261106_180441.jpg'},
                            use_cache=True
                            )


image_path = r"C:\Users\acer\Documents\MEGA\requ\pics\20261106_180446.jpg"
image = cv.imread(image_path)

predictions = results[0]['predictions']['predictions']

for predict in predictions:
    x = int(predict['x'])
    y = int(predict['y'])
    
    width = int(predict['width'])
    height = int(predict['height'])
    lable = predict['class']
    confidence = predict["confidence"]
    
    x1 = x - width // 2
    y1 = y - height // 2
    x2 = x + width // 2
    y2 = y + height // 2
    
    cv.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 4)
    cv.putText(image, f'{lable} {confidence:.2}', (x1, y1 - 50), cv.FONT_HERSHEY_DUPLEX, 1.80, (0,0,255), 7)
    
cv.imwrite('detection.jpg', image)
print('saved: detection.jpg') 
    
