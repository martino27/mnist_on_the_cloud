from flask import Flask, redirect, render_template, request, url_for
from flask_wtf.file import FileField
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form, ValidationError
import forward
import backward
import io
import base64
import tempfile

# Patch the location of gfile
tf.gfile = tf.io.gfile

app = Flask(__name__)

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())


def is_image():

  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image


class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])


def encode_image(image):
    image_buffer = io.BytesIO()
    image.save(image_buffer, format='PNG')
    imgstr = 'data:image/png;base64,{:s}'.format(
        base64.b64encode(image_buffer.getvalue()).decode().replace("'", "")
    )
    return imgstr


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found!")
                return -1


def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready


def detect_num(testPic):
    image = Image.open(testPic)
    testPicArr = pre_pic(testPic)
    preValue = restore_model(testPicArr)
    result = {}
    result['image'] = encode_image(image.copy())
    result['detectNum'] = preValue
    return result


@app.route("/")
def upload():
    photo_form = PhotoForm(request.form)
    return render_template('upload.html', photo_form=photo_form, result={})


@app.route("/post", methods=['POST', 'GET'])
def post():
    form = PhotoForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST' and form.validate():
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            form.input_photo.data.save(temp)
            temp.flush()
            result = detect_num(temp.name)
            temp.close()

        photo_form = PhotoForm(request.form)
        return render_template('upload.html',
                               photo_form=photo_form, result=result)
    else:
        return redirect(url_for('upload'))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8848, debug=True)