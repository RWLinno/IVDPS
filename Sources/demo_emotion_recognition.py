# In[20]:
import urllib.request
import urllib.error
import time
import re
import json
import jsonpath


# In[21]:


# In[22]:
class Emotion_Recognition_v1():

    def __init__(self, filepath):
        self.http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
        self.key = "N7biI8dANjx6bAKO9EHKzeMRVBc3rqND"
        self.secret = "84EJ1kpxaqhs1uGRdCMp7pw5QeqL9Xt9"
        self.filepath = filepath

    def recognize_emotion(self):
        boundary = '----------%s' % hex(int(time.time() * 1000))
        data = []
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
        data.append(self.key)
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
        data.append(self.secret)
        data.append('--%s' % boundary)
        fr = open(self.filepath, 'rb')
        data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
        data.append('Content-Type: %s\r\n' % 'application/octet-stream')
        data.append(fr.read())
        fr.close()
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
        data.append('1')
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
        data.append(
            "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
        data.append('--%s--\r\n' % boundary)


        # In[23]:
        for i, d in enumerate(data):
            if isinstance(d, str):
                data[i] = d.encode('utf-8')

        http_body = b'\r\n'.join(data)

        # build http request
        req = urllib.request.Request(url=self.http_url, data=http_body)

        # header
        req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        qrcont = resp.read()
        # if you want to load as json, you should decode first,
        # for example: json.loads(qrount.decode('utf-8'))
        obj = json.loads(qrcont.decode('utf-8'))
        return obj

    # In[ ]:




