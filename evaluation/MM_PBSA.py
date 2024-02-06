import os
import parmed as pmd
from parmed.amber import AmberParm, MMinitialize, MMPBSA
# from mmtools import AmberParm, MMinitialize, MMPBSA

# 将pdb文件转化为ParmEd结构对象
protein_structure = pmd.load_file('./data/test_data/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10.pdb')
# 将sdf文件转化为ParmEd结构对象
ligand_structure = pmd.load_file('./logs/Final_crossdock_pocket_revisedCA_velEGNN_no_fix_2022_12_16__13_05_02/generalized_fix_nodes_partial_build_500pt_result_2022_12_16__20_04_17/generated_ligand_all/-9.1_4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10_gen.sdf')

# 将蛋白质和配体结构写入pdb文件和mol2文件
protein_structure.write_pdb('protein.pdb')
ligand_structure.write_mol2('ligand.mol2')

# 创建Amber参数文件
os.system('antechamber -i ligand.mol2 -fi mol2 -o ligand.mol2 -fo mol2 -c bcc -s 2')
os.system('parmchk -i ligand.mol2 -f mol2 -o frcmod.mol2')

# 使用LEaP创建top和crd文件
top_file = 'p_l.top'
crd_file = 'p_l.inpcrd'
with open('leap.in', 'w') as f:
    f.write(f"""source leaprc.gaff2
    loadamberparams frcmod.mol2
    mol = loadpdb protein.pdb
    ligand = loadmol2 ligand.mol2
    combine = combine{{mol ligand}}
    savepdb combine go.pdb
    saveamberparm combine {top_file} {crd_file}
    quit""")
os.system('tleap -f leap.in')

# 读入top和crd文件
system = AmberParm(top_file, crd_file)
trajectory = MMinitialize.read_traj(crd_file)

# 设置MM-PBSA计算选项
options = {'solvation' : 'gb', 'use_sander' : False}

# 进行MM-PBSA计算
mm_pbsa = MMPBSA(system, options=options)
binding_energy, breakdown = mm_pbsa(trajectory)

# 输出结果
print('蛋白质-配体结合自由能为：{:.2f} kcal/mol'.format(binding_energy))
print('能量贡献项包括：')
for key, value in breakdown.items():
    print(key + ': ' + str(value))