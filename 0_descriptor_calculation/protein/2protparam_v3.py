import os
import argparse
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv

# Define three-letter abbreviations for amino acids
amino_acid_3_letter = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
}

# Define three-letter abbreviations for atoms
atomic_3_letter = {
    'C': 'Carbon', 'H': 'Hydrogen', 'N': 'Nitrogen', 'O': 'Oxygen', 'S': 'Sulfur'
}

# Manually build atomic_composition
atomic_composition = {
    'A': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 0},  # Alanine
    'R': {'C': 6, 'H': 14, 'N': 4, 'O': 2, 'S': 0},  # Arginine
    'N': {'C': 4, 'H': 8, 'N': 2, 'O': 3, 'S': 0},  # Asparagine
    'D': {'C': 4, 'H': 7, 'N': 1, 'O': 4, 'S': 0},  # Aspartic acid
    'C': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1},  # Cysteine
    'E': {'C': 5, 'H': 9, 'N': 1, 'O': 4, 'S': 0},  # Glutamic acid
    'Q': {'C': 5, 'H': 10, 'N': 2, 'O': 3, 'S': 0},  # Glutamine
    'G': {'C': 2, 'H': 5, 'N': 1, 'O': 2, 'S': 0},  # Glycine
    'H': {'C': 6, 'H': 9, 'N': 3, 'O': 2, 'S': 0},  # Histidine
    'I': {'C': 6, 'H': 13, 'N': 1, 'O': 2, 'S': 0},  # Isoleucine
    'L': {'C': 6, 'H': 13, 'N': 1, 'O': 2, 'S': 0},  # Leucine
    'K': {'C': 6, 'H': 14, 'N': 2, 'O': 2, 'S': 0},  # Lysine
    'M': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1},  # Methionine
    'F': {'C': 9, 'H': 11, 'N': 1, 'O': 2, 'S': 0},  # Phenylalanine
    'P': {'C': 5, 'H': 9, 'N': 1, 'O': 2, 'S': 0},  # Proline
    'S': {'C': 3, 'H': 7, 'N': 1, 'O': 3, 'S': 0},  # Serine
    'T': {'C': 4, 'H': 9, 'N': 1, 'O': 3, 'S': 0},  # Threonine
    'W': {'C': 11, 'H': 12, 'N': 2, 'O': 2, 'S': 0},  # Tryptophan
    'Y': {'C': 9, 'H': 11, 'N': 1, 'O': 3, 'S': 0},  # Tyrosine
    'V': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 0},  # Valine
}

# Calculate half-life
def calculate_half_life(protein_sequence):
    half_life_table = {
        'A': {'Mammalian': '4.4', 'Yeast': '>20', 'E. coli': '>10'},  # Ala
        'R': {'Mammalian': '1.0', 'Yeast': '2', 'E. coli': '2'},     # Arg
        'N': {'Mammalian': '1.4', 'Yeast': '3', 'E. coli': '>10'},   # Asn
        'D': {'Mammalian': '1.1', 'Yeast': '3', 'E. coli': '>10'},   # Asp
        'C': {'Mammalian': '1.2', 'Yeast': '>20', 'E. coli': '>10'}, # Cys
        'Q': {'Mammalian': '0.8', 'Yeast': '10', 'E. coli': '>10'},  # Gln
        'E': {'Mammalian': '1.0', 'Yeast': '30', 'E. coli': '>10'},  # Glu
        'G': {'Mammalian': '30.0', 'Yeast': '>20', 'E. coli': '>10'},# Gly
        'H': {'Mammalian': '3.5', 'Yeast': '10', 'E. coli': '>10'},  # His
        'I': {'Mammalian': '20.0', 'Yeast': '30', 'E. coli': '>10'}, # Ile
        'L': {'Mammalian': '5.5', 'Yeast': '3', 'E. coli': '2'},     # Leu
        'K': {'Mammalian': '1.3', 'Yeast': '3', 'E. coli': '2'},     # Lys
        'M': {'Mammalian': '30.0', 'Yeast': '>20', 'E. coli': '>10'},# Met
        'F': {'Mammalian': '1.1', 'Yeast': '3', 'E. coli': '2'},     # Phe
        'P': {'Mammalian': '>20', 'Yeast': '>20', 'E. coli': '?'},   # Pro
        'S': {'Mammalian': '1.9', 'Yeast': '>20', 'E. coli': '>10'}, # Ser
        'T': {'Mammalian': '7.2', 'Yeast': '>20', 'E. coli': '>10'}, # Thr
        'W': {'Mammalian': '2.8', 'Yeast': '3', 'E. coli': '2'},     # Trp
        'Y': {'Mammalian': '2.8', 'Yeast': '10', 'E. coli': '2'},    # Tyr
        'V': {'Mammalian': '100.0', 'Yeast': '>20', 'E. coli': '>10'}# Val
    }
    
    n_terminal_aa = protein_sequence[0]
    half_life = half_life_table.get(n_terminal_aa, None)
    
    if half_life is None:
        return f"Unknown N-terminal amino acid: {n_terminal_aa}"
    
    return n_terminal_aa, half_life


# Read FASTA file and analyze sequences
def analyze_fasta(input_file, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        # Define column names in the specified order, with corresponding prefixes
        fieldnames = [
            'protein',
            'protein_Protparam_Molecular_weight',
            'protein_Protparam_Atomic_composition_C',
            'protein_Protparam_Atomic_composition_H',
            'protein_Protparam_Atomic_composition_N',
            'protein_Protparam_Atomic_composition_O',
            'protein_Protparam_Atomic_composition_S',
            'protein_Protparam_Atomic_composition_total',
            'protein_Protparam_OC_Ext_coefficient',
            'protein_Protparam_OC_Abs',
            'protein_Protparam_RC_Ext_coefficient',
            'protein_Protparam_RC_Abs',
            'protein_Protparam_Instability_index',
            'protein_Protparam_Aliphatic_index',
            'protein_Protparam_GRAVY',
            'protein_Protparam_Analysis_Aromaticity',
            'protein_Protparam_Analysis_SSF_1',
            'protein_Protparam_Analysis_SSF_2',
            'protein_Protparam_Analysis_SSF_3',
            'protein_Protparam_Analysis_Isoelectric_point'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Get the file name of the FASTA file
        protein_name = os.path.splitext(input_file.split('/')[-1])[0]

        for record in SeqIO.parse(input_file, 'fasta'):
            protein_sequence = str(record.seq)
            prot_param = ProteinAnalysis(protein_sequence)
            
            molecular_weight = prot_param.molecular_weight()
            isoelectric_point = prot_param.isoelectric_point()
            count_aa = prot_param.count_amino_acids()
            total_aa = sum(count_aa.values())
            molar_extinction_coefficient = prot_param.molar_extinction_coefficient()
            instability_index = prot_param.instability_index()
            gravy = prot_param.gravy()
            aromaticity = prot_param.aromaticity()
            secondary_structure_fraction = prot_param.secondary_structure_fraction()
            
            # Calculate the mole percentage of each amino acid
            percent_aa = {aa: (count / total_aa) * 100 for aa, count in count_aa.items()}
            
            # Calculate the aliphatic index
            aliphatic_index = (
                percent_aa.get('A', 0) +  # Mole percentage of alanine (Ala)
                2.9 * percent_aa.get('V', 0) +  # Mole percentage of valine (Val) times 2.9
                3.9 * (percent_aa.get('I', 0) + percent_aa.get('L', 0))  # Mole percentage of isoleucine (Ile) and leucine (Leu) times 3.9
            )
            
            # Calculate the molecular formula
            molecular_formula = {
                'C': 0, 'H': 0, 'N': 0, 'O': 0, 'S': 0
            }
            for aa, count in count_aa.items():
                for atom, num in atomic_composition[aa].items():
                    molecular_formula[atom] += num * count
            
            # Account for the loss of water molecules during peptide bond formation
            num_peptide_bonds = len(protein_sequence) - 1
            molecular_formula['H'] -= 2 * num_peptide_bonds
            molecular_formula['O'] -= num_peptide_bonds
            
            atomic_composition_total = sum(molecular_formula.values())
            
            # Write to the CSV file, with keys matching the new fieldnames
            writer.writerow({
                'protein': protein_name,
                'protein_Protparam_Molecular_weight': molecular_weight,
                'protein_Protparam_Atomic_composition_C': molecular_formula['C'],
                'protein_Protparam_Atomic_composition_H': molecular_formula['H'],
                'protein_Protparam_Atomic_composition_N': molecular_formula['N'],
                'protein_Protparam_Atomic_composition_O': molecular_formula['O'],
                'protein_Protparam_Atomic_composition_S': molecular_formula['S'],
                'protein_Protparam_Atomic_composition_total': atomic_composition_total,
                'protein_Protparam_OC_Ext_coefficient': molar_extinction_coefficient[1],
                'protein_Protparam_OC_Abs': molar_extinction_coefficient[1] / molecular_weight,
                'protein_Protparam_RC_Ext_coefficient': molar_extinction_coefficient[0],
                'protein_Protparam_RC_Abs': molar_extinction_coefficient[0] / molecular_weight,
                'protein_Protparam_Instability_index': instability_index,
                'protein_Protparam_Aliphatic_index': aliphatic_index,
                'protein_Protparam_GRAVY': gravy,
                'protein_Protparam_Analysis_Aromaticity': aromaticity,
                'protein_Protparam_Analysis_SSF_1': secondary_structure_fraction[0],
                'protein_Protparam_Analysis_SSF_2': secondary_structure_fraction[1],
                'protein_Protparam_Analysis_SSF_3': secondary_structure_fraction[2],
                'protein_Protparam_Analysis_Isoelectric_point': isoelectric_point
            })



# Main function
def main():
    parser = argparse.ArgumentParser(description='Analyze protein sequences from a FASTA file.')
    parser.add_argument('-i', '--input', required=True, help='Input FASTA file')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file')
    args = parser.parse_args()

    analyze_fasta(args.input, args.output)

if __name__ == '__main__':
    main()