"""Ejecuta en orden los scripts de los ejercicios 1 a 7 del laboratorio."""
import p1_load_integrate
import p2_quality_cleaning
import p3_eda
import p4_bipartite_network
import p5_network_projections
import p6_topology_fragmentation
import p7_communities


def main():
    print("\n########## EJERCICIO 1: Carga e integracion ##########")
    p1_load_integrate.main()

    print("\n########## EJERCICIO 2: Calidad y limpieza ##########")
    p2_quality_cleaning.main()

    print("\n########## EJERCICIO 3: Analisis exploratorio ##########")
    p3_eda.main()

    print("\n########## EJERCICIO 4: Red bipartita autor-video ##########")
    p4_bipartite_network.main()

    print("\n########## EJERCICIO 5: Proyecciones ##########")
    p5_network_projections.main()

    print("\n########## EJERCICIO 6: Topologia y fragmentacion ##########")
    p6_topology_fragmentation.main()

    print("\n########## EJERCICIO 7: Comunidades ##########")
    p7_communities.main()


if __name__ == "__main__":
    main()
