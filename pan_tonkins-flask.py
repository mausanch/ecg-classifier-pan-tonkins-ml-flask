from distutils.log import debug
from flask import abort, Flask, jsonify, request
from flask import render_template, render_template_string, redirect
import logging.config
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler

import numpy
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from flask_cors import CORS, cross_origin

from datetime import datetime,timedelta

scaler = MinMaxScaler()

# CONFIGURACIÓN DE VARIABLE PARA LA CREACIÓN DE SERVICIOS REST
service = Flask(__name__)
cors = CORS(service, resources={r"/*": {"origins": "*"}})

service.config['CORS_HEADERS'] = 'Content-Type'


# CONFIGURACIÓN DE LOG PARA PINTAR EL FLUJO
# logging.config.dictConfig(LOGGING)
log = logging.getLogger("mswitch")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-1s %(filename)-20s %(lineno)-4d %(levelname)-8s %(message)s')

fileHandler = ConcurrentRotatingFileHandler(
    'logs', maxBytes=1000000,
    backupCount=10)


handler.setFormatter(formatter)
log.addHandler(handler)
log.addHandler(fileHandler)

log.setLevel(logging.DEBUG)

# METODO QUE LLAMA AL SCRIPT
@service.route('/', methods=['POST'])
@cross_origin()
def procesamiento_ecg():
    log.info('Procesamiento Iniciado')
    log.info("JSON RECIBIDO")
    log.info(request.form)
    log.info(request.headers)

    request_data = request.get_json()

    request_data['']

    resiñtado =2
    
    return resultado


if __name__ == '__main__':
    service.run(host="0.0.0.0", port=8091, debug=True)
