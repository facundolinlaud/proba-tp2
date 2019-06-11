import statistics

# Ejercicio 1
## Momentos de la muestra
def estimar_b_mom_distr_uniforme(a, muestra):
	# Tomamos la media de la muestra, la multiplicamos por dos y la restamos por el parámetro 'a'
	# de la distribución normal, que en nuestro caso es 0.
	return 2 * statistics.mean(muestra) - a

## Estimador de Máxima Verosimilitud
def estimar_b_mv_distr_uniforme(muestra):
	# Retornamos el valor máximo dentro de nuestra muestra.
	return max(muestra)

# Ejercicio 2
def estimador_doble_mediana(muestra):
	# Obtenemos la mediana de la muestra, la multiplicamos por 2 y retornamos el resultado.
	return 2 * statistics.median(muestra)