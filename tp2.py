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
	print(f"  Con la muestra: {muestra}")
	print(f"  b_mom = {b_mom} y su error es {b - b_mom}")
	print(f"  b_mv = {b_mv} y su error es {b - b_mv}")
	print(f"  b_med = {b_med} y su error es {b - b_med}")

#################################################################################################
####################################### Ejercicio 4 #############################################
#################################################################################################

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
	sesgo = estimador - b

	# Ahora calculamos la varianza muestral del estimador. La función statistics.variance() es la
	# varianza muestral que provee Python mientras que statistics.pvariance() es la varianza poblacional.
	# En este caso, es adecuado utilizar statistics.variance(). Utilizaremos nuestra propia implementacion
	# de todas maneras.
	# var_estimador = statistics.variance(muestra, estimador)

	var_estimador = calcular_varianza_muestral(muestra, estimador)

	# Finalmente, procedemos a calcular el Error Cuadrático Medio del estimador utilizando su varianza
	# muestral y su respectivo sesgo:
	ecm = var_estimador + (sesgo ** 2)

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
		"ecm": ecm,
		"b": b
	}

def calcular_varianza_muestral(muestra, esperanza):
	varianza = 0

	# Sumo a la varianza el cuadrado de cada valor en la muestra menos el promedio de la muestra
	for valor in muestra:
		varianza += (valor - esperanza)**2

	# Luego retorno la sumatoria dividida por n - 1
	return varianza / (len(muestra) - 1)

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
	step = 0.01
	bs = numpy.arange(start = a + step, stop = b, step = step)

	simulaciones_mv = [simulacion_mv(b = b, n = 15) for b in bs]
	simulaciones_mom = [simulacion_mom(b = b, n = 15) for b in bs]
	simulaciones_med = [simulacion_med(b = b, n = 15) for b in bs]

	# pdb.set_trace()
	sesgos_mv = como_lista("sesgo", simulaciones_mv)
	sesgos_mv_porcentaje_err = [sesgos_mv[i] * 100 / bs[i] for i in range(len(simulaciones_mv))]
	varianzas_mv = como_lista("var_estimador", simulaciones_mv)
	ecms_mv = como_lista("ecm", simulaciones_mv)

	sesgos_mom = como_lista("sesgo", simulaciones_mom)
	sesgos_mom_porcentaje_err = [sesgos_mom[i] * 100 / bs[i] for i in range(len(simulaciones_mom))]
	varianzas_mom = como_lista("var_estimador", simulaciones_mom)
	ecms_mom = como_lista("ecm", simulaciones_mom)

	sesgos_med = como_lista("sesgo", simulaciones_med)
	sesgos_med_porcentaje_err = [sesgos_med[i] * 100 / bs[i] for i in range(len(simulaciones_med))]
	varianzas_med = como_lista("var_estimador", simulaciones_med)
	ecms_med = como_lista("ecm", simulaciones_med)

	leyendas = ["Maxima Verosimilitud", "Momento", "Doble Mediana"]


	# Agregados
	setup_plot("Sesgos (\% de diferencia con b)", bs, [sesgos_mv_porcentaje_err], 
		labels=[leyendas[0]], graficar_grande=(6, 6), xlabel="b", marker='')
	plt.ylim((-20,20))
	show_plot(output="sesgos-mv-porcentaje-err.png", save_instead_of_plotting = False)

	setup_plot("Sesgos (\% de diferencia con b)", bs, [sesgos_mom_porcentaje_err], 
		labels=[leyendas[1]], graficar_grande=(6, 6), xlabel="b", marker='')
	plt.ylim((-20,20))
	show_plot(output="sesgos-mom-porcentaje-err.png", save_instead_of_plotting = False)

	setup_plot("Sesgos (\% de diferencia con b)", bs, [sesgos_med_porcentaje_err], 
		labels=[leyendas[2]], graficar_grande=(6, 6), xlabel="b", marker='')
	plt.ylim((-20,20))
	show_plot(output="sesgos-med-porcentaje-err.png", save_instead_of_plotting = False)


	# Defaults
	plot("Sesgos", bs, [sesgos_mv, sesgos_mom, sesgos_med], 
		labels=leyendas, graficar_grande=True, xlabel="b", output="sesgos.png",
		save_instead_of_plotting = False, marker='')

	plot("Varianzas", bs, [varianzas_mv, varianzas_mom, varianzas_med], 
		labels=leyendas, graficar_grande=True, xlabel="b", output="varianzas.png",
		save_instead_of_plotting = False, marker='')

	plot("ECM", bs, [ecms_mv, ecms_mom, ecms_med], 
		labels=leyendas, graficar_grande=True, xlabel="b", output="ecm.png",
		save_instead_of_plotting = False, marker='')

#################################################################################################
####################################### Ejercicio 7 #############################################
#################################################################################################

def plot_varianzas_estimadores(simulaciones_mv, simulaciones_mom, simulaciones_med):
	var_mv = statistics.mean([simulacion['var_estimador'] for simulacion in simulaciones_mv])
	var_mom = statistics.mean([simulacion['var_estimador'] for simulacion in simulaciones_mom])
	var_med = statistics.mean([simulacion['var_estimador'] for simulacion in simulaciones_med])

	plt.axhline(y=var_mv, color='b', linewidth=0.8, linestyle='dotted')
	plt.axhline(y=var_mom, color='orange', linewidth=0.8, linestyle='dotted')
	plt.axhline(y=var_med, color='g', linewidth=0.8, linestyle='dotted')
	plt.plot([], [], color='black', linestyle='dotted', linewidth=0.8, label=r"$V(\hat{\theta})$")

def plot_x_axis():
	plt.axhline(0, color='black', linewidth=1)

def ejercicio_7():
	ns = [15, 30, 60, 120, 240, 480]#, 960, 1920, 2500, 3000, 3500, 4000]

	simulaciones_mv = [simulacion_mv(b = 1.0, n = n) for n in ns]
	simulaciones_mom = [simulacion_mom(b = 1.0, n = n) for n in ns]
	simulaciones_med = [simulacion_med(b = 1.0, n = n) for n in ns]

	# ECM en función de n
	ns_ecm_mv = como_lista("ecm", simulaciones_mv)
	ns_ecm_mom = como_lista("ecm", simulaciones_mom)
	ns_ecm_med = como_lista("ecm", simulaciones_med)

	leyendas = ["Maxima Verosimilitud", "Momento", "Doble Mediana"]

	setup_plot("ECM", ns, [ns_ecm_mv, ns_ecm_mom, ns_ecm_med], 
		labels=leyendas, graficar_grande = False, xlabel="n", marker='.')
	plot_x_axis()
	plot_varianzas_estimadores(simulaciones_mv, simulaciones_mom, simulaciones_med)
	show_plot(output="ecm-en-f-de-n.png",
		save_instead_of_plotting = False)

	# Sesgos en función de n
	ns_sesgos_mv = como_lista("sesgo", simulaciones_mv)
	ns_sesgos_mom = como_lista("sesgo", simulaciones_mom)
	ns_sesgos_med = como_lista("sesgo", simulaciones_med)

	setup_plot("Sesgo", ns, [ns_sesgos_mv, ns_sesgos_mom, ns_sesgos_med],
		labels=leyendas, graficar_grande = False, xlabel="n", marker='.')
	plot_x_axis()
	show_plot(output="sesgos-en-f-de-n.png", save_instead_of_plotting = False)

	# Varianza en función de n
	ns_var_mv = como_lista("var_estimador", simulaciones_mv)
	ns_var_mom = como_lista("var_estimador", simulaciones_mom)
	ns_var_med = como_lista("var_estimador", simulaciones_med)

	setup_plot("Varianza", ns, [ns_var_mv, ns_var_mom, ns_var_med],
		labels=leyendas, graficar_grande = False, xlabel="n", marker='.')
	plot_x_axis()
	show_plot(output="varianzas-en-f-de-n-small.png", save_instead_of_plotting = False)

#################################################################################################
####################################### Ejercicio 8 #############################################
#################################################################################################

def ejercicio_8():
	# Tomamos la muestra dada por la catedra.
	muestra = [0.917, 0.247, 0.384, 0.530, 0.798, 0.912, 0.096, 0.684, 0.394, 20.1, 0.769, 0.137, 0.352, 0.332, 0.670]

	b_mom = estimar_b_por_momentos(0, muestra)
	b_med = estimar_b_por_doble_mediana(muestra)
	b_mv = estimar_b_por_maxima_verosimilitud(muestra)

	print(f"Ejercicio 8")
	print(f"	Estimador por momentos: {b_mom}")
	print(f"	Estimador por maxima verosimilitud: {b_mv}")
	print(f"	Estimador por doble mediana: {b_med}")

#################################################################################################
####################################### Ejercicio 9 #############################################
#################################################################################################

def ejercicio_9():
	a = 0
	b = 1
	Nrep = 1000

	muestra_b_moms = []
	muestra_b_mvs = []
	muestra_b_meds = []

	for _ in range(Nrep): 
		#tomamos una muestra aleatoria X_1, X_2, ..., X_15, con X_i ~ U[0, 1].
		m = obtener_muestra_uniforme(0, 1, 15)

		#De forma independiente, tenemos que cada elemento se multiplica por 100 con proba 0.005.
		#Entonces, hacemos 15 muestras Bernoulli.
		m_bernoulli = numpy.random.binomial(1, 0.005, 15)

		#Tomamos entonces tomamos la muestra uniforme y la modificamos segun la muestra Bernoulli
		for i in range(15):
			if m_bernoulli[i] == 1:
				m[i] *= 100

		b_mom = estimar_b_por_momentos(0, m)
		b_mv = estimar_b_por_maxima_verosimilitud(m)
		b_med = estimar_b_por_doble_mediana(m)

		muestra_b_moms.append(b_mom)
		muestra_b_mvs.append(b_mv)
		muestra_b_meds.append(b_med)

	print(f"Ejercicio 9")
	simular_sesgo_varianza_y_ecm_para_estimador("Momentos", b = b, muestra = muestra_b_moms)
	simular_sesgo_varianza_y_ecm_para_estimador("Máxima Verosimilitud", b = b, muestra = muestra_b_mvs)
	simular_sesgo_varianza_y_ecm_para_estimador("Doble Mediana", b = b, muestra = muestra_b_meds)


#################################################################################################
########################################## MISC #################################################
#################################################################################################

def plot(titulo_eje_y, xs, yss, labels, graficar_grande, xlabel, output, save_instead_of_plotting, marker):
	setup_plot(titulo_eje_y, xs, yss, labels, graficar_grande, xlabel, marker)
	show_plot(output, save_instead_of_plotting)

def como_lista(parametro, simulaciones):
	return [simulacion[parametro] for simulacion in simulaciones]

def setup_plot(titulo_eje_y, xs, yss, labels, graficar_grande, xlabel, marker):	
	if graficar_grande == True:
		# plt.rcParams["figure.figsize"] = (16, 4)
		plt.figure(figsize=(16, 4)) 
	elif graficar_grande == False:
		# plt.rcParams["figure.figsize"] = (8, 6)
		plt.figure(figsize=(8, 6)) 
	else:
		plt.figure(figsize=graficar_grande) 

	plt.tight_layout(pad=0)
	plt.xlabel(xlabel)
	plt.ylabel(titulo_eje_y)

	for ys, label in zip(yss, labels):
		plt.plot(xs, ys, label=label, linewidth=0.8, marker=marker)
	
	plt.grid(True)
	plt.grid(b=True, which='major', color='black', linestyle='dotted', alpha=0.1)
	plt.grid(b=True, which='minor', color='black', linestyle='dotted', alpha=0.05)
	plt.minorticks_on()

def show_plot(output, save_instead_of_plotting):
	plt.legend()

	if not save_instead_of_plotting:
		plt.show()
	else:
		plt.savefig(sys.path[0] + "/informe/imagenes/" + output, dpi=160, bbox_inches='tight')

def ejecutar_ejercicios():
	ejercicio_3()
	ejercicio_4()
	ejercicio_6()
	ejercicio_7()
	ejercicio_8()
	ejercicio_9()

ejecutar_ejercicios()