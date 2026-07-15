class CircuitBreaker:
    def __init__(self, balance_inicial=10000.0, limite_drawdown=0.25, max_posicion_pct=0.20):
        self.limite_drawdown = limite_drawdown
        self.max_posicion_pct = max_posicion_pct
        self.balance_inicial = balance_inicial 

    def set_balance_inicial(self, nuevo_balance):
        self.balance_inicial = float(nuevo_balance)
        print(f"Circuit Breaker configurado con balance inicial de: {self.balance_inicial}")

    def es_seguro_operar(self, balance_actual, net_worth, cantidad_a_invertir):
        # Validación de Drawdown
        drawdown = (self.balance_inicial - net_worth) / self.balance_inicial
        if drawdown > self.limite_drawdown:
            print("Drawdown excedido. Operativa bloqueada por seguridad.")
            return False

        # Validación de Capital
        if cantidad_a_invertir > (balance_actual * self.max_posicion_pct):
            print("Intento de inversión excesivo. Bloqueado.")
            return False

        return True