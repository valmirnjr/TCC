import cupyx.profiler as profiler

def cpr_profile(sigRx, paramCPR, prec, loops):
    results = profiler.benchmark(
        carrierRecovery.cpr, # Funcao avaliada
        (sigRx,), # Sinal recebido
        {
            "paramCPR": paramCPR, # Parametros do algoritmo CPR
            "prec": prec # Precisao em ponto flutuante
        },
        n_repeat=loops, # Numero de repeticoes
        n_warmup=3, # Numero de execuções antes de começar a medir
        name=paramCPR.alg # Nome de algoritmo
    )
    return results
