import csv
import os
from datetime import datetime

class OperacionesLogger:
    def __init__(self, filename="operaciones_por_agente.csv"):
        self.filename = filename
        # Si no existe, creamos la cabecera
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Fecha", "Accion", "Precio", "Cantidad", "Estado", "Razon"])

    def registrar(self, accion, precio, cantidad, estado, razon):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                             accion, precio, cantidad, estado, razon])