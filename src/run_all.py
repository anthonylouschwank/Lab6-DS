"""Ejecuta en orden los scripts de los ejercicios 1 a 4 (avance del laboratorio)."""
import p1_load_integrate
import p2_quality_cleaning
import p3_eda
import p4_bipartite_network


def main():
    print("\n########## EJERCICIO 1: Carga e integracion ##########")
    p1_load_integrate.main()

    print("\n########## EJERCICIO 2: Calidad y limpieza ##########")
    p2_quality_cleaning.main()

    print("\n########## EJERCICIO 3: Analisis exploratorio ##########")
    p3_eda.main()

    print("\n########## EJERCICIO 4: Red bipartita autor-video ##########")
    p4_bipartite_network.main()


if __name__ == "__main__":
    main()
