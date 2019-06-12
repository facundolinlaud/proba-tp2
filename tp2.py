# Para correr el trabajo práctico es necesario invocar "python3 tp.py" con versión de python 3

import sys
import statistics
import numpy
import matplotlib.pyplot as plt
import pdb

#################################################################################################
####################################### Ejercicio 1 #############################################
#################################################################################################

## Momentos de la muestra
def estimar_b_por_momentos(a, muestra):
	# Tomamos la media de la muestra, la multiplicamos por dos y la restamos por el parámetro 'a'
	# de la distribución normal, que en nuestro caso es 0.
	return 2 * statistics.mean(muestra) - a

## Estimador de Máxima Verosimilitud
def estimar_b_por_maxima_verosimilitud(muestra):
	# Retornamos el valor máximo dentro de nuestra muestra.
	return max(muestra)

#################################################################################################
####################################### Ejercicio 2 #############################################
#################################################################################################

def estimar_b_por_doble_mediana(muestra):
	# Obtenemos la mediana de la muestra, la multiplicamos por 2 y retornamos el resultado.
	return 2 * statistics.median(muestra)

# Ejercicio 3
def obtener_muestra_uniforme(a, b, cantidad):
	# numpy.random devuelve 'cantidad' de reales equiprobables entre [a, b)
	return numpy.random.uniform(low=a, high=b, size=cantidad)

#################################################################################################
####################################### Ejercicio 3 #############################################
#################################################################################################

def ejercicio_3():
	print("Ejercicio 3:")

	# El enunciado nos propone utilizar [a, b) = [0, 1)
	a = 0
	b = 1
	muestra = obtener_muestra_uniforme(a, b, cantidad = 15)

	# Obtenemos el estimador de momentos, el estimador de máxima verosimilitud y el de la doble mediana de b.
	b_mom = estimar_b_por_momentos(a, muestra)
	b_mv = estimar_b_por_maxima_verosimilitud(muestra)
	b_med = estimar_b_por_doble_mediana(muestra)

	# Obtenemos el error restando los estimadores al valor verdadero de 'b' que es 2.
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# !!!!!!! TO DO: CHECKEAR SI ESTO ESTA BIEN !!!!!!!!!!!!!!!
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	print(f"  Con la muestra: {muestra}")
	print(f"  b_mom = {b_mom} y su error es {b - b_mom}")
	print(f"  b_mv = {b_mv} y su error es {b - b_mv}")
	print(f"  b_med = {b_med} y su error es {b - b_med}")

#################################################################################################
####################################### Ejercicio 4 #############################################
#################################################################################################

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!! TO DO: HAY QUE JUSTIFICAR EN LATEX ESTE EJERCICIO? !!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ejercicio_4():
	print("Ejercicio 4")
	# El enunciado nos propone utilizar [a, b) = [0, 1)
	a = 0
	b = 1
	Nrep = 1000

	muestra_b_moms = []
	muestra_b_mvs = []
	muestra_b_meds = []

	# Calculamos cada estimador Nrep (= 1000) veces utilizando la misma muestra para cada iteración
	# (para luego comparar los estimadores) y ponemos los resultado en listas.
	for _ in range(Nrep):
		muestra_uniforme = obtener_muestra_uniforme(a, b, cantidad = 15)
		muestra_b_moms.append(estimar_b_por_momentos(a, muestra_uniforme))
		muestra_b_mvs.append(estimar_b_por_maxima_verosimilitud(muestra_uniforme))
		muestra_b_meds.append(estimar_b_por_doble_mediana(muestra_uniforme))

	# Esta función se llama para la muestra de cada estimador. Realiza lo que pide el enunciado
	# sin tener que duplicar el código para cada estimador.
	simular_sesgo_varianza_y_ecm_para_estimador("Momentos", b = b, muestra = muestra_b_moms)
	simular_sesgo_varianza_y_ecm_para_estimador("Máxima Verosimilitud", b = b, muestra = muestra_b_mvs)
	simular_sesgo_varianza_y_ecm_para_estimador("Doble Mediana", b = b, muestra = muestra_b_meds)

def simular_sesgo_varianza_y_ecm_para_estimador(nombre_estimador, b, muestra):
	# Calculamos el promedio del estimador invocando la función statistics.mean sobre la muestra
	estimador = statistics.mean(muestra)

	# Calculamos el sesgo del estimador restándolo con el valor real de 'b'
	sesgo = b - estimador

	# Ahora calculamos la varianza muestral del estimador. La función statistics.variance() es la
	# varianza muestral que provee Python mientras que statistics.pvariance() es la varianza poblacional.
	# En este caso, es adecuado utilizar statistics.variance():
	var_estimador = statistics.variance(muestra, estimador)

	# Finalmente, procedemos a calcular el Error Cuadrático Medio del estimador utilizando su varianza
	# muestral y su respectivo sesgo:
	ecm = var_estimador + sesgo ** 2

	if nombre_estimador != None: # Porque no siempre voy a querer mostrar todo esto por pantalla
		print(f"	Simulando estimador {nombre_estimador}")
		print(f"		Promedio del estimador: {estimador}")
		print(f"		Sesgo: {sesgo}")
		print(f"		Varianza del estimador: {var_estimador}")
		print(f"		ECM del estimador: {ecm}")

	# Retorno lo calculado para utilizarlo en el ejercicio 5
	return {
		"estimador": estimador,
		"sesgo": sesgo,
		"var_estimador": var_estimador,
		"ecm": ecm
	}

#################################################################################################
####################################### Ejercicio 5 #############################################
#################################################################################################

def simulacion_mv(b, n):
	muestra_estimadores = [estimar_b_por_maxima_verosimilitud(
		obtener_muestra_uniforme(a = 0, b = b, cantidad = 15)) for _ in range(n)]

	return simular_sesgo_varianza_y_ecm_para_estimador(None, b, muestra_estimadores)


def simulacion_mom(b, n):
	muestra_estimadores = [estimar_b_por_momentos(a = 0,
		muestra = obtener_muestra_uniforme(a = 0, b = b, cantidad = 15)) for _ in range(n)]

	return simular_sesgo_varianza_y_ecm_para_estimador(None, b, muestra_estimadores)

def simulacion_med(b, n):
	muestra_estimadores = [estimar_b_por_doble_mediana(
		obtener_muestra_uniforme(a = 0, b = b, cantidad = 15)) for _ in range(n)]

	return simular_sesgo_varianza_y_ecm_para_estimador(None, b, muestra_estimadores)

#################################################################################################
####################################### Ejercicio 6 #############################################
#################################################################################################

def ejercicio_6():
	a = 0.0
	b = 2.0
	step = 0.1
	bs = numpy.arange(start = a + step, stop = b, step = step)

	simulaciones_mv = [simulacion_mv(b = b, n = 15) for b in bs]
	simulaciones_mom = [simulacion_mom(b = b, n = 15) for b in bs]
	simulaciones_med = [simulacion_med(b = b, n = 15) for b in bs]

	sesgos_mv = como_lista("sesgo", simulaciones_mv)
	varianzas_mv = como_lista("var_estimador", simulaciones_mv)
	ecms_mv = como_lista("ecm", simulaciones_mv)

	sesgos_mom = como_lista("sesgo", simulaciones_mom)
	varianzas_mom = como_lista("var_estimador", simulaciones_mom)
	ecms_mom = como_lista("ecm", simulaciones_mom)

	sesgos_med = como_lista("sesgo", simulaciones_med)
	varianzas_med = como_lista("var_estimador", simulaciones_med)
	ecms_med = como_lista("ecm", simulaciones_med)

	leyendas = ["Maxima Verosimilitud", "Momento", "Doble Mediana"]
	plot("Sesgos", bs, [sesgos_mv, sesgos_mom, sesgos_med], leyendas, graficar_grande=True)
	plot("Varianzas", bs, [varianzas_mv, varianzas_mom, varianzas_med], leyendas, graficar_grande=True)
	plot("ECM", bs, [ecms_mv, ecms_mom, ecms_med], leyendas, graficar_grande=True)

def como_lista(parametro, simulaciones):
	return [simulacion[parametro] for simulacion in simulaciones]

def plot(titulo_eje_y, xs, yss, labels, graficar_grande):
	if graficar_grande:
		plt.rcParams["figure.figsize"] = (16, 4)

	plt.tight_layout(pad=0)
	plt.xlabel("Valores de b")
	plt.ylabel(titulo_eje_y)

	for ys, label in zip(yss, labels):
		plt.plot(xs, ys, label=label)
	
	plt.grid(True)
	plt.grid(b=True, which='major', color='black', linestyle='dotted', alpha=0.1)
	plt.grid(b=True, which='minor', color='black', linestyle='dotted', alpha=0.05)
	plt.minorticks_on()
	plt.legend()
	plt.draw()
	plt.show()
	# plt.savefig(sys.path[0] + "/informe/imagenes/" + titulo_eje_y.lower() + ".png", dpi=160, bbox_inches='tight')
	plt.clf()

#################################################################################################
########################################### MISC ################################################
#################################################################################################

def ejecutar_ejercicios():
	ejercicio_3()
	ejercicio_4()
	ejercicio_6()

ejecutar_ejercicios()