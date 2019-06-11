import statistics

# Ejercicio 1
## Momentos de la muestra
def estimar_b_mom_distr_uniforme(a, muestra):
	return 2 * statistics.mean(muestra) - a

## Estimador de Máxima Verosimilitud
def estimar_b_mv_distr_uniforme(muestra):
	return max(muestra)