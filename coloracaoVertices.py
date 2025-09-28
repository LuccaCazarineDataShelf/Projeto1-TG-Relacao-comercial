from grafoMatriz import GrafoMatrizND

hospital = GrafoMatrizND(9)
hospital.insereA(0, 5)  # A–F
hospital.insereA(0, 6)  # A–G
hospital.insereA(0, 8)  # A–I
hospital.insereA(1, 2)  # B–C
hospital.insereA(1, 7)  # B–H
hospital.insereA(2, 4)  # C–E
hospital.insereA(2, 7)  # C–H
hospital.insereA(3, 4)  # D–E
hospital.insereA(3, 5)  # D–F
hospital.insereA(3, 7)  # D–H
hospital.insereA(4, 6)  # E–G
hospital.insereA(5, 6)  # F–G
hospital.insereA(5, 8)  # F–I
hospital.insereA(6, 7)  # G–H
hospital.insereA(6, 8)  # G–I

hospital.show()
quartos = hospital.coloracao()
print(quartos)