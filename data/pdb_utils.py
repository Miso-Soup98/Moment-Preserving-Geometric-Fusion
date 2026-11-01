from copy import copy, deepcopy
import math
import os
from typing import Dict, List, Tuple
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom
from configs import IMGT, Chothia

class AminoAcid:

    def __init__(self, symbol: str, abrv: str, sidechain: List[str], idx=0):
        self.symbol = symbol
        self.abrv = abrv
        self.idx = idx
        self.sidechain = sidechain

    def __str__(self):
        return '{0} {1} {2} {3}'.format(self.idx, self.symbol, self.abrv, self.sidechain)

class AminoAcidVocab:
    MAX_ATOM_NUMBER = 14

    def __init__(self):
        self.backbone_atoms = ['N', 'CA', 'C', 'O']
        self.PAD, self.MASK = ('#', '*')
        self.BOA, self.BOH, self.BOL = ('&', '+', '-')
        specials = [(self.PAD, 'PAD'), (self.MASK, 'MASK'), (self.BOA, '<X>'), (self.BOH, '<H>'), (self.BOL, '<L>')]
        aas = [('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'), ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'), ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'), ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'), ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO')]
        self.atom_pad, self.atom_mask = ('p', 'm')
        self.atom_pos_mask, self.atom_pos_bb, self.atom_pos_pad = ('m', 'b', 'p')
        sidechain_map = {'G': [], 'A': ['CB'], 'V': ['CB', 'CG1', 'CG2'], 'L': ['CB', 'CG', 'CD1', 'CD2'], 'I': ['CB', 'CG1', 'CG2', 'CD1'], 'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'], 'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'], 'D': ['CB', 'CG', 'OD1', 'OD2'], 'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'], 'N': ['CB', 'CG', 'OD1', 'ND2'], 'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'], 'K': ['CB', 'CG', 'CD', 'CE', 'NZ'], 'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'], 'M': ['CB', 'CG', 'SD', 'CE'], 'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'], 'S': ['CB', 'OG'], 'T': ['CB', 'OG1', 'CG2'], 'C': ['CB', 'SG'], 'P': ['CB', 'CG', 'CD']}
        self.chi_angles_atoms = {'ALA': [], 'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']], 'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], 'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], 'CYS': [['N', 'CA', 'CB', 'SG']], 'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']], 'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']], 'GLY': [], 'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']], 'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']], 'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']], 'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']], 'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']], 'SER': [['N', 'CA', 'CB', 'OG']], 'THR': [['N', 'CA', 'CB', 'OG1']], 'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'VAL': [['N', 'CA', 'CB', 'CG1']]}
        self.sidechain_bonds = {'ALA': {'CA': ['CB']}, 'GLY': {}, 'VAL': {'CA': ['CB'], 'CB': ['CG1', 'CG2']}, 'LEU': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD2', 'CD1']}, 'ILE': {'CA': ['CB'], 'CB': ['CG1', 'CG2'], 'CG1': ['CD1']}, 'MET': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['SD'], 'SD': ['CE']}, 'PHE': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD1', 'CD2'], 'CD1': ['CE1'], 'CD2': ['CE2'], 'CE1': ['CZ']}, 'TRP': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD1', 'CD2'], 'CD1': ['NE1'], 'CD2': ['CE2', 'CE3'], 'CE2': ['CZ2'], 'CZ2': ['CH2'], 'CE3': ['CZ3']}, 'PRO': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD']}, 'SER': {'CA': ['CB'], 'CB': ['OG']}, 'THR': {'CA': ['CB'], 'CB': ['OG1', 'CG2']}, 'CYS': {'CA': ['CB'], 'CB': ['SG']}, 'TYR': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD1', 'CD2'], 'CD1': ['CE1'], 'CD2': ['CE2'], 'CE1': ['CZ'], 'CZ': ['OH']}, 'ASN': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['OD1', 'ND2']}, 'GLN': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD'], 'CD': ['OE1', 'NE2']}, 'ASP': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['OD1', 'OD2']}, 'GLU': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD'], 'CD': ['OE1', 'OE2']}, 'LYS': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD'], 'CD': ['CE'], 'CE': ['NZ']}, 'ARG': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['CD'], 'CD': ['NE'], 'NE': ['CZ'], 'CZ': ['NH1', 'NH2']}, 'HIS': {'CA': ['CB'], 'CB': ['CG'], 'CG': ['ND1', 'CD2'], 'ND1': ['CE1'], 'CD2': ['NE2']}}
        _all = aas + specials
        self.amino_acids = [AminoAcid(symbol, abrv, sidechain_map.get(symbol, [])) for symbol, abrv in _all]
        self.symbol2idx, self.abrv2idx = ({}, {})
        for i, aa in enumerate(self.amino_acids):
            self.symbol2idx[aa.symbol] = i
            self.abrv2idx[aa.abrv] = i
            aa.idx = i
        self.special_mask = [0 for _ in aas] + [1 for _ in specials]
        self.idx2atom = [self.atom_pad, self.atom_mask, 'C', 'N', 'O', 'S']
        self.idx2atom_pos = [self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_bb, 'B', 'G', 'D', 'E', 'Z', 'H']
        self.atom2idx, self.atom_pos2idx = ({}, {})
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i

    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.amino_acids[idx].symbol

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.amino_acids[idx].abrv

    def abrv_to_idx(self, abrv):
        abrv = abrv.upper()
        return self.abrv2idx.get(abrv, None)

    def symbol_to_idx(self, symbol):
        symbol = symbol.upper()
        return self.symbol2idx.get(symbol, None)

    def idx_to_symbol(self, idx):
        return self.amino_acids[idx].symbol

    def idx_to_abrv(self, idx):
        return self.amino_acids[idx].abrv

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)

    def get_special_mask(self):
        return copy(self.special_mask)

    def get_atom_type_mat(self):
        atom_pad = self.get_atom_pad_idx()
        mat = []
        for i, aa in enumerate(self.amino_acids):
            atoms = [atom_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:
                atom_mask = self.get_atom_mask_idx()
                atoms = [atom_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                for aidx, atom in enumerate(self.backbone_atoms + aa.sidechain):
                    atoms[aidx] = self.atom_to_idx(atom[0])
            mat.append(atoms)
        return mat

    def get_atom_pos_mat(self):
        atom_pos_pad = self.get_atom_pos_pad_idx()
        mat = []
        for i, aa in enumerate(self.amino_acids):
            aps = [atom_pos_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:
                atom_pos_mask = self.get_atom_pos_mask_idx()
                aps = [atom_pos_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                aidx = 0
                for _ in self.backbone_atoms:
                    aps[aidx] = self.atom_pos_to_idx(self.atom_pos_bb)
                    aidx += 1
                for atom in aa.sidechain:
                    aps[aidx] = self.atom_pos_to_idx(atom[1])
                    aidx += 1
            mat.append(aps)
        return mat

    def get_sidechain_info(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return copy(self.amino_acids[idx].sidechain)

    def get_sidechain_geometry(self, symbol):
        abrv = self.symbol_to_abrv(symbol)
        chi_angles_atoms = copy(self.chi_angles_atoms[abrv])
        sidechain_bonds = self.sidechain_bonds[abrv]
        return (chi_angles_atoms, sidechain_bonds)

    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]

    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]

    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]

    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]

    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx[atom_pos]

    def get_num_atom_type(self):
        return len(self.idx2atom)

    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_amino_acid_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)
VOCAB = AminoAcidVocab()

def format_aa_abrv(abrv):
    if abrv == 'MSE':
        return 'MET'
    return abrv

class Residue:

    def __init__(self, symbol: str, coordinate: Dict, _id: Tuple):
        self.symbol = symbol
        self.coordinate = coordinate
        self.sidechain = VOCAB.get_sidechain_info(symbol)
        self.id = _id

    def get_symbol(self):
        return self.symbol

    def get_coord(self, atom_name):
        return copy(self.coordinate[atom_name])

    def get_coord_map(self) -> Dict[str, List]:
        return deepcopy(self.coordinate)

    def get_backbone_coord_map(self) -> Dict[str, List]:
        coord = {atom: self.coordinate[atom] for atom in self.coordinate if atom in VOCAB.backbone_atoms}
        return coord

    def get_sidechain_coord_map(self) -> Dict[str, List]:
        coord = {}
        for atom in self.sidechain:
            if atom in self.coordinate:
                coord[atom] = self.coordinate[atom]
        return coord

    def get_atom_names(self):
        return list(self.coordinate.keys())

    def get_id(self):
        return self.id

    def set_symbol(self, symbol):
        assert VOCAB.symbol_to_abrv(symbol) is not None, '{0} is not an amino acid'.format(symbol)
        self.symbol = symbol

    def set_coord(self, coord):
        self.coordinate = deepcopy(coord)

    def dist_to(self, residue):
        xa = np.array(list(self.get_coord_map().values()))
        xb = np.array(list(residue.get_coord_map().values()))
        if len(xa) == 0 or len(xb) == 0:
            return math.nan
        dist = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
        return np.min(dist)

    def to_bio(self):
        _id = (' ', self.id[0], self.id[1])
        residue = BResidue(_id, VOCAB.symbol_to_abrv(self.symbol), '    ')
        atom_map = self.coordinate
        for i, atom in enumerate(atom_map):
            fullname = ' ' + atom
            while len(fullname) < 4:
                fullname += ' '
            bio_atom = BAtom(name=atom, coord=np.array(atom_map[atom], dtype=np.float32), bfactor=0, occupancy=1.0, altloc=' ', fullname=fullname, serial_number=i, element=atom[0])
            residue.add(bio_atom)
        return residue

    def __iter__(self):
        return iter([(atom_name, self.coordinate[atom_name]) for atom_name in self.coordinate])

class Peptide:

    def __init__(self, _id, residues: List[Residue]):
        self.residues = residues
        self.seq = ''
        self.id = _id
        for residue in residues:
            self.seq += residue.get_symbol()

    def set_id(self, _id):
        self.id = _id

    def get_id(self):
        return self.id

    def get_seq(self):
        return self.seq

    def get_span(self, i, j):
        i, j = (max(i, 0), min(j, len(self.seq)))
        if j <= i:
            return None
        else:
            residues = deepcopy(self.residues[i:j])
            return Peptide(self.id, residues)

    def get_residue(self, i):
        return deepcopy(self.residues[i])

    def get_ca_pos(self, i):
        return copy(self.residues[i].get_coord('CA'))

    def get_cb_pos(self, i):
        return copy(self.residues[i].get_coord('CB'))

    def set_residue_coord(self, i, coord):
        self.residues[i].set_coord(coord)

    def set_residue_translation(self, i, vec):
        coord = self.residues[i].get_coord_map()
        for atom in coord:
            ori_vec = coord[atom]
            coord[atom] = [a + b for a, b in zip(ori_vec, vec)]
        self.set_residue_coord(i, coord)

    def set_residue_symbol(self, i, symbol):
        self.residues[i].set_symbol(symbol)
        self.seq = self.seq[:i] + symbol + self.seq[i + 1:]

    def set_residue(self, i, symbol, coord):
        self.set_residue_symbol(i, symbol)
        self.set_residue_coord(i, coord)

    def to_bio(self):
        chain = BChain(id=self.id)
        for residue in self.residues:
            chain.add(residue.to_bio())
        return chain

    def __iter__(self):
        return iter(self.residues)

    def __len__(self):
        return len(self.seq)

    def __str__(self):
        return self.seq

class Protein:

    def __init__(self, pdb_id, peptides):
        self.pdb_id = pdb_id
        self.peptides = peptides

    @classmethod
    def from_pdb(cls, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb_path)
        pdb_id = structure.header['idcode'].upper().strip()
        if pdb_id == '':
            pdb_id = os.path.split(pdb_path)[1].split('.')[0] + '(filename)'
        peptides = {}
        for chain in structure.get_chains():
            _id = chain.get_id()
            residues = []
            has_non_residue = False
            for residue in chain:
                abrv = residue.get_resname()
                hetero_flag, res_number, insert_code = residue.get_id()
                if hetero_flag != ' ':
                    continue
                symbol = VOCAB.abrv_to_symbol(abrv)
                if symbol is None:
                    has_non_residue = True
                    break
                atoms = {atom.get_id(): atom.get_coord() for atom in residue if atom.element != 'H'}
                residues.append(Residue(symbol, atoms, (res_number, insert_code)))
            if has_non_residue or len(residues) == 0:
                continue
            peptides[_id] = Peptide(_id, residues)
        return cls(pdb_id, peptides)

    def get_id(self):
        return self.pdb_id

    def num_chains(self):
        return len(self.peptides)

    def get_chain(self, name):
        if name in self.peptides:
            return deepcopy(self.peptides[name])
        else:
            return None

    def get_chain_names(self):
        return list(self.peptides.keys())

    def to_bio(self):
        structure = BStructure(id=self.pdb_id)
        model = BModel(id=0)
        for name in self.peptides:
            model.add(self.peptides[name].to_bio())
        structure.add(model)
        return structure

    def to_pdb(self, path, atoms=None):
        if atoms is None:
            bio_structure = self.to_bio()
        else:
            prot = deepcopy(self)
            for _, chain in prot:
                for residue in chain:
                    coordinate = {}
                    for atom in atoms:
                        if atom in residue.coordinate:
                            coordinate[atom] = residue.coordinate[atom]
                    residue.coordinate = coordinate
            bio_structure = prot.to_bio()
        io = PDBIO()
        io.set_structure(bio_structure)
        io.save(path)

    def __iter__(self):
        return iter([(c, self.peptides[c]) for c in self.peptides])

    def __eq__(self, other):
        if not isinstance(other, Protein):
            raise TypeError('Cannot compare other type to Protein')
        for key in self.peptides:
            if key in other.peptides and self.peptides[key].seq == other.peptides[key].seq:
                continue
            else:
                return False
        return True

    def __str__(self):
        res = self.pdb_id + '\n'
        for seq_name in self.peptides:
            res += '\t{0}: {1}\n'.format(seq_name, self.peptides[seq_name])
        return res

class AgAbComplex:
    num_interface_residues = 48

    def __init__(self, antigen: Protein, antibody: Protein, heavy_chain: str, light_chain: str, numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False) -> None:
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.numbering = numbering
        self.antigen = antigen
        if skip_validity_check:
            self.antibody, self.cdr_pos = (antibody, None)
        else:
            self.antibody, self.cdr_pos = self._extract_antibody_info(antibody, numbering)
        self.pdb_id = antigen.get_id()
        if skip_epitope_cal:
            self.epitope = None
        else:
            self.epitope = self._cal_epitope()

    @classmethod
    def from_pdb(cls, pdb_path: str, heavy_chain: str, light_chain: str, antigen_chains: List[str], numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
        protein = Protein.from_pdb(pdb_path)
        pdb_id = protein.get_id()
        ab_peptides = {heavy_chain: protein.get_chain(heavy_chain), light_chain: protein.get_chain(light_chain)}
        ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None}
        for chain in antigen_chains:
            assert chain in ag_peptides, 'Antigen chain {0} has something wrong!'.format(chain)
        antigen = Protein(pdb_id, ag_peptides)
        antibody = Protein(pdb_id, ab_peptides)
        return cls(antigen, antibody, heavy_chain, light_chain, numbering, skip_epitope_cal, skip_validity_check)

    def _extract_antibody_info(self, antibody: Protein, numbering: str):
        numbering = numbering.lower()
        if numbering == 'imgt':
            _scheme = IMGT
        elif numbering.lower() == 'chothia':
            _scheme = Chothia
        else:
            raise NotImplementedError('Numbering scheme {0} not implemented'.format(numbering))
        h_type_mapping, l_type_mapping = ({}, {})
        for lo, hi in [_scheme.HFR1, _scheme.HFR2, _scheme.HFR3, _scheme.HFR4]:
            for i in range(lo, hi + 1):
                h_type_mapping[i] = '0'
        for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.H1, _scheme.H2, _scheme.H3]):
            for i in range(lo, hi + 1):
                h_type_mapping[i] = cdr
        h_conserved = _scheme.Hconserve
        for lo, hi in [_scheme.LFR1, _scheme.LFR2, _scheme.LFR3, _scheme.LFR4]:
            for i in range(lo, hi + 1):
                l_type_mapping[i] = '0'
        for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.L1, _scheme.L2, _scheme.L3]):
            for i in range(lo, hi + 1):
                l_type_mapping[i] = cdr
        l_conserved = _scheme.Lconserve
        selected_peptides, cdr_pos = ({}, {})
        for c, chain_name in zip(['H', 'L'], [self.heavy_chain, self.light_chain]):
            chain = antibody.get_chain(chain_name)
            assert chain is not None, 'Chain {0} not found in the antibody'.format(chain_name)
            type_mapping = h_type_mapping if c == 'H' else l_type_mapping
            conserved = h_conserved if c == 'H' else l_conserved
            res_type = ''
            for i in range(len(chain)):
                residue = chain.get_residue(i)
                residue_number = residue.get_id()[0]
                if residue_number in type_mapping:
                    res_type += type_mapping[residue_number]
                    if residue_number in conserved:
                        hit, symbol = (False, residue.get_symbol())
                        for conserved_residue in conserved[residue_number]:
                            if symbol == VOCAB.abrv_to_symbol(conserved_residue):
                                hit = True
                                break
                        assert hit, 'Not {0} at {1}'.format(conserved[residue_number], residue_number)
                else:
                    res_type += '-'
            if '0' not in res_type:
                print(self.heavy_chain, self.light_chain, antibody.pdb_id, res_type)
            start, end = (res_type.index('0'), res_type.rindex('0'))
            for cdr in ['1', '2', '3']:
                cdr_start, cdr_end = (res_type.find(cdr), res_type.rfind(cdr))
                assert cdr_start != -1, 'cdr {0}{1} not found, residue type: {2}'.format(c, cdr, res_type)
                start, end = (min(start, cdr_start), max(end, cdr_end))
                cdr_pos['CDR-{0}{1}'.format(c, cdr)] = (cdr_start, cdr_end)
            for cdr in ['1', '2', '3']:
                cdr = 'CDR-{0}{1}'.format(c, cdr)
                cdr_start, cdr_end = cdr_pos[cdr]
                cdr_pos[cdr] = (cdr_start - start, cdr_end - start)
            chain = chain.get_span(start, end + 1)
            chain.set_id(chain_name)
            selected_peptides[chain_name] = chain
        antibody = Protein(antibody.get_id(), selected_peptides)
        return (antibody, cdr_pos)

    def _cal_epitope(self):
        ag_rids, ag_xs, ab_xs = ([], [], [])
        ag_mask, ab_mask = ([], [])
        cdrh3 = self.get_cdr('H3')
        for _type, protein in zip(['ag', 'ab'], [self.antigen, [('A', cdrh3)]]):
            is_ag = _type == 'ag'
            rids = []
            if is_ag:
                xs, masks = (ag_xs, ag_mask)
            else:
                xs, masks = (ab_xs, ab_mask)
            for chain_name, chain in protein:
                for i, residue in enumerate(chain):
                    bb_coord = residue.get_backbone_coord_map()
                    sc_coord = residue.get_sidechain_coord_map()
                    coord = {}
                    coord.update(bb_coord)
                    coord.update(sc_coord)
                    num_pad = VOCAB.MAX_ATOM_NUMBER - len(coord)
                    x = [coord[key] for key in coord] + [[0, 0, 0] for _ in range(num_pad)]
                    mask = [1 for _ in coord] + [0 for _ in range(num_pad)]
                    rids.append((chain_name, i))
                    xs.append(x)
                    masks.append(mask)
            if is_ag:
                ag_rids = rids
        assert len(ag_xs) != 0, 'No antigen structure!'
        ag_xs, ab_xs = (np.array(ag_xs), np.array(ab_xs))
        ag_mask, ab_mask = (np.array(ag_mask).astype('bool'), np.array(ab_mask).astype('bool'))
        dist = np.linalg.norm(ag_xs[:, None] - ab_xs[None, :], axis=-1)
        dist = dist + np.logical_not(ag_mask[:, None] * ab_mask[None, :]) * 1000000.0
        min_dists = np.min(np.min(dist, axis=-1), axis=-1)
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return epitope

    def get_id(self) -> str:
        return self.antibody.pdb_id

    def get_antigen(self) -> Protein:
        return deepcopy(self.antigen)

    def get_epitope(self, cdrh3_pos=None) -> List[Tuple[Residue, str, int]]:
        if cdrh3_pos is not None:
            backup = self.cdr_pos
            self.cdr_pos = {'CDR-H3': [cdrh3_pos[0], cdrh3_pos[1]]}
            epitope = self._cal_epitope()
            self.cdr_pos = backup
            return deepcopy(epitope)
        if self.epitope is None:
            self.epitope = self._cal_epitope()
        return deepcopy(self.epitope)

    def get_interacting_residues(self, dist_cutoff=5) -> Tuple[List[int], List[int]]:
        ag_rids, ag_xs, ab_xs = ([], [], [])
        for chain_name in self.antigen.get_chain_names():
            chain = self.antigen.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:
                    continue
                ag_rids.append((chain_name, i))
                ag_xs.append(x)
        for chain_name in self.antibody.get_chain_names():
            chain = self.antibody.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:
                    continue
                ab_xs.append(x)
        assert len(ag_xs) != 0, 'No antigen structure!'
        ag_xs, ab_xs = (np.array(ag_xs), np.array(ab_xs))
        dist = np.linalg.norm(ag_xs[:, None, :] - ab_xs[None, :, :], axis=-1)
        min_dists = np.min(dist, axis=1)
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return

    def get_heavy_chain(self) -> Peptide:
        return self.antibody.get_chain(self.heavy_chain)

    def get_light_chain(self) -> Peptide:
        return self.antibody.get_chain(self.light_chain)

    def get_framework(self, fr):
        seg_id = int(fr[-1])
        chain = self.get_heavy_chain() if fr[0] == 'H' else self.get_light_chain()
        begin, end = (-1, -1)
        if seg_id == 1:
            begin, end = (0, self.get_cdr_pos(fr[0] + str(seg_id))[0])
        elif seg_id == 4:
            begin, end = (self.get_cdr_pos(fr[0] + '3')[-1] + 1, len(chain))
        else:
            begin = self.get_cdr_pos(fr[0] + str(seg_id - 1))[-1] + 1
            end = self.get_cdr_pos(fr[0] + str(seg_id))[0]
        return chain.get_span(begin, end)

    def get_cdr_pos(self, cdr='H3'):
        cdr = 'CDR-{0}'.format(cdr).upper()
        if cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        else:
            return None

    def get_cdr(self, cdr='H3'):
        cdr = cdr.upper()
        pos = self.get_cdr_pos(cdr)
        if pos is None:
            return None
        chain = self.get_heavy_chain() if 'H' in cdr else self.get_light_chain()
        return chain.get_span(pos[0], pos[1] + 1)

    def to_pdb(self, path, atoms=None):
        peptides = {}
        for name in self.antigen.get_chain_names():
            peptides[name] = self.antigen.get_chain(name)
        for name in self.antibody.get_chain_names():
            peptides[name] = self.antibody.get_chain(name)
        protein = Protein(self.get_id(), peptides)
        protein.to_pdb(path, atoms)

    def __str__(self):
        pdb_info = 'PDB ID: {0}'.format(self.pdb_id)
        antibody_info = 'Antibody H-{0} ({1}), '.format(self.heavy_chain, len(self.get_heavy_chain())) + 'L-{0} ({1})'.format(self.light_chain, len(self.get_light_chain()))
        antigen_info = 'Antigen Chains: {0}'.format([(ag, len(self.antigen.get_chain(ag))) for ag in self.antigen.get_chain_names()])
        cdr_info = 'CDRs: \n'.format()
        for name in self.cdr_pos:
            chain = self.get_heavy_chain() if 'H' in name else self.get_light_chain()
            start, end = self.cdr_pos[name]
            cdr_info += '\t{0}: [{1}, {2}], {3}\n'.format(name, start, end, chain.seq[start:end + 1])
        epitope_info = 'Epitope: \n'.format()
        residue_map = {}
        for _, chain_name, i in self.get_epitope():
            if chain_name not in residue_map:
                residue_map[chain_name] = []
            residue_map[chain_name].append(i)
        for chain_name in residue_map:
            epitope_info += '\t{0}: {1}\n'.format(chain_name, sorted(residue_map[chain_name]))
        sep = '\n' + '=' * 20 + '\n'
        return sep + pdb_info + '\n' + antibody_info + '\n' + cdr_info + '\n' + antigen_info + '\n' + epitope_info + sep

def merge_to_one_chain(protein: Protein):
    residues = []
    chain_order = sorted(protein.get_chain_names())
    for chain_name in chain_order:
        chain = protein.get_chain(chain_name)
        for _, residue in enumerate(chain.residues):
            residue.id = (len(residues), ' ')
            residues.append(residue)
    return Protein(protein.get_id(), {'A': Peptide('A', residues)})
