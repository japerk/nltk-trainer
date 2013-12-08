# ----------------------------------------------------------
# AdvaS Advanced Search 
# module for phonetic algorithms
#
# (C) 2002 - 2005 Frank Hofmann, Chemnitz, Germany
# email fh@efho.de
# ----------------------------------------------------------

# changed 2005-01-24

import string
import re

def soundex (term):
	"Return the soundex value to a string argument."

	# Create and compare soundex codes of English words.
	#
	# Soundex is an algorithm that hashes English strings into
	# alpha-numerical value that represents what the word sounds
	# like. For more information on soundex and some notes on the
	# differences in implemenations visit:
	# http://www.bluepoof.com/Soundex/info.html
	#
	# This version modified by Nathan Heagy at Front Logic Inc., to be
	# compatible with php's soundexing and much faster.
	#
	# eAndroid / Nathan Heagy / Jul 29 2000
	# changes by Frank Hofmann / Jan 02 2005

	# generate translation table only once. used to translate into soundex numbers
	#table = string.maketrans('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123012002245501262301020201230120022455012623010202')
	table = string.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', '01230120022455012623010202')

	# check parameter
	if not term:
		return "0000" # could be Z000 for compatibility with other implementations
	# end if

		# convert into uppercase letters
	term = string.upper(term)
	first_char = term[0]

	# translate the string into soundex code according to the table above
	term = string.translate(term[1:], table)
	
	# remove all 0s
	term = string.replace(term, "0", "")
	# remove duplicate numbers in-a-row
	str2 = first_char
	for x in term:
		if x != str2[-1]:
			str2 = str2 + x
		# end if
	# end for

	# pad with zeros
	str2 = str2+"0"*len(str2)

	# take the first four letters
	return_value = str2[:4]

	# return value
	return return_value

def metaphone (term):
	"returns metaphone code for a given string"

	# implementation of the original algorithm from Lawrence Philips
	# extended/rewritten by M. Kuhn
	# improvements with thanks to John Machin <sjmachin@lexicon.net>

	# define return value
	code = ""

	i = 0
	term_length = len(term)

	if (term_length == 0):
		# empty string ?
		return code
	# end if

	# extension #1 (added 2005-01-28)
	# convert to lowercase
	term = string.lower(term)
	
	# extension #2 (added 2005-01-28)
	# remove all non-english characters, first
	term = re.sub(r'[^a-z]', '', term)
	if len(term) == 0:
		# nothing left
		return code
	# end if
		
	# extension #3 (added 2005-01-24)
	# conflate repeated letters
	firstChar = term[0]
	str2 = firstChar
	for x in term:
		if x != str2[-1]:
			str2 = str2 + x
		# end if
	# end for
	
	# extension #4 (added 2005-01-24)
	# remove any vowels unless a vowel is the first letter
	firstChar = str2[0]
	str3 = firstChar
	for x in str2[1:]:
		if (re.search(r'[^aeiou]', x)):
			str3 = str3 + x
		# end if
	# end for
	
	term = str3
	term_length = len(term)
	if term_length == 0:
		# nothing left
		return code
	# end if
	
	# check for exceptions
	if (term_length > 1):
		# get first two characters
		first_chars = term[0:2]

		# build translation table
		table = {
			"ae":"e",
			"gn":"n",
			"kn":"n",
			"pn":"n",
			"wr":"n",
			"wh":"w"
		}
		
		if first_chars in table.keys():
			term = term[2:]
			code = table[first_chars]
			term_length = len(term)
		# end if
		
	elif (term[0] == "x"):
		term = ""
		code = "s"
		term_length = 0
	# end if

	# define standard translation table
	st_trans = {
		"b":"b",
		"c":"k",
		"d":"t",
		"g":"k",
		"h":"h",
		"k":"k",
		"p":"p",
		"q":"k",
		"s":"s",
		"t":"t",
		"v":"f",
		"w":"w",
		"x":"ks",
		"y":"y",
		"z":"s"
	}

	i = 0
	while (i<term_length):
		# init character to add, init basic patterns
		add_char = ""
		part_n_2 = ""
		part_n_3 = ""
		part_n_4 = ""
		part_c_2 = ""
		part_c_3 = ""

		# extract a number of patterns, if possible
		if (i < (term_length - 1)):
			part_n_2 = term[i:i+2]

			if (i>0):
				part_c_2 = term[i-1:i+1]
				part_c_3 = term[i-1:i+2]
			# end if
		# end if

		if (i < (term_length - 2)):
			part_n_3 = term[i:i+3]
		# end if

		if (i < (term_length - 3)):
			part_n_4 = term[i:i+4]
		# end if

		# use table with conditions for translations
		if (term[i] == "b"):
			add_char = st_trans["b"]
			if (i == (term_length - 1)):
				if (i>0):
					if (term[i-1] == "m"):
						add_char = ""
					# end if
				# end if
			# end if
		elif (term[i] == "c"):
			add_char = st_trans["c"]
			if (part_n_2 == "ch"):
				add_char = "x"
			elif (re.search(r'c[iey]', part_n_2)):
				add_char = "s"
			# end if

			if (part_n_3 == "cia"):
				add_char = "x"
			# end if

			if (re.search(r'sc[iey]', part_c_3)):
				add_char = ""
			# end if

		elif (term[i] == "d"):
			add_char = st_trans["d"]
			if (re.search(r'dg[eyi]', part_n_3)):
				add_char = "j"
			# end if

		elif (term[i] == "g"):
			add_char = st_trans["g"]

			if (part_n_2 == "gh"):
				if (i == (term_length - 2)):
					add_char = ""
				# end if
			elif (re.search(r'gh[aeiouy]', part_n_3)):
				add_char = ""
			elif (part_n_2 == "gn"):
				add_char = ""
			elif (part_n_4 == "gned"):
				add_char = ""
			elif (re.search(r'dg[eyi]',part_c_3)):
				add_char = ""
			elif (part_n_2 == "gi"):
				if (part_c_3 != "ggi"):
					add_char = "j"
				# end if
			elif (part_n_2 == "ge"):
				if (part_c_3 != "gge"):
					add_char = "j"
				# end if
			elif (part_n_2 == "gy"):
				if (part_c_3 != "ggy"):
					add_char = "j"
				# end if
			elif (part_n_2 == "gg"):
				add_char = ""
			# end if
		elif (term[i] == "h"):
			add_char = st_trans["h"]
			if (re.search(r'[aeiouy]h[^aeiouy]', part_c_3)):
				add_char = ""
			elif (re.search(r'[csptg]h', part_c_2)):
				add_char = ""
			# end if
		elif (term[i] == "k"):
			add_char = st_trans["k"]
			if (part_c_2 == "ck"):
				add_char = ""
			# end if
		elif (term[i] == "p"):
			add_char = st_trans["p"]
			if (part_n_2 == "ph"):
				add_char = "f"
			# end if
		elif (term[i] == "q"):
			add_char = st_trans["q"]
		elif (term[i] == "s"):
			add_char = st_trans["s"]
			if (part_n_2 == "sh"):
				add_char = "x"
			# end if

			if (re.search(r'si[ao]', part_n_3)):
				add_char = "x"
			# end if
		elif (term[i] == "t"):
			add_char = st_trans["t"]
			if (part_n_2 == "th"):
				add_char = "0"
			# end if

			if (re.search(r'ti[ao]', part_n_3)):
				add_char = "x"
			# end if
		elif (term[i] == "v"):
			add_char = st_trans["v"]
		elif (term[i] == "w"):
			add_char = st_trans["w"]
			if (re.search(r'w[^aeiouy]', part_n_2)):
				add_char = ""
			# end if
		elif (term[i] == "x"):
			add_char = st_trans["x"]
		elif (term[i] == "y"):
			add_char = st_trans["y"]
		elif (term[i] == "z"):
			add_char = st_trans["z"]
		else:
			# alternative
			add_char = term[i]
		# end if

		code = code + add_char
		i += 1
	# end while

	# return metaphone code
	return code

def nysiis (term):
	"returns New York State Identification and Intelligence Algorithm (NYSIIS) code for the given term"

	code = ""

	i = 0
	term_length = len(term)

	if (term_length == 0):
		# empty string ?
		return code
	# end if

	# build translation table for the first characters
	table = {
		"mac":"mcc",
		"ph":"ff",
		"kn":"nn",
		"pf":"ff",
		"k":"c",
		"sch":"sss"
	}

	for table_entry in table.keys():
		table_value = table[table_entry]	# get table value
		table_value_len = len(table_value)	# calculate its length
		first_chars = term[0:table_value_len]
		if (first_chars == table_entry):
			term = table_value + term[table_value_len:]
			break
		# end if
	# end for

	# build translation table for the last characters
	table = {
		"ee":"y",
		"ie":"y",
		"dt":"d",
		"rt":"d",
		"rd":"d",
		"nt":"d",
		"nd":"d",
	}

	for table_entry in table.keys():
		table_value = table[table_entry]	# get table value
		table_entry_len = len(table_entry)	# calculate its length
		last_chars = term[(0 - table_entry_len):]
		#print last_chars, ", ", table_entry, ", ", table_value
		if (last_chars == table_entry):
			term = term[:(0 - table_value_len + 1)] + table_value
			break
		# end if
	# end for

	# initialize code
	code = term

	# transform ev->af
	code = re.sub(r'ev', r'af', code)

	# transform a,e,i,o,u->a
	code = re.sub(r'[aeiouy]', r'a', code)
	
	# transform q->g
	code = re.sub(r'q', r'g', code)
	
	# transform z->s
	code = re.sub(r'z', r's', code)

	# transform m->n
	code = re.sub(r'm', r'n', code)

	# transform kn->n
	code = re.sub(r'kn', r'n', code)

	# transform k->c
	code = re.sub(r'k', r'c', code)

	# transform sch->sss
	code = re.sub(r'sch', r'sss', code)

	# transform ph->ff
	code = re.sub(r'ph', r'ff', code)

	# transform h-> if previous or next is nonvowel -> previous
	occur = re.findall(r'([a-z]{0,1}?)h([a-z]{0,1}?)', code)
	#print occur
	for occur_group in occur:
		occur_item_previous = occur_group[0]
		occur_item_next = occur_group[1]

		if ((re.match(r'[^aeiouy]', occur_item_previous)) or (re.match(r'[^aeiouy]', occur_item_next))):
			if (occur_item_previous != ""):
				# make substitution
				code = re.sub (occur_item_previous + "h", occur_item_previous * 2, code, 1)
			# end if
		# end if
	# end for
	
	# transform w-> if previous is vowel -> previous
	occur = re.findall(r'([aeiouy]{1}?)w', code)
	#print occur
	for occur_group in occur:
		occur_item_previous = occur_group[0]
		# make substitution
		code = re.sub (occur_item_previous + "w", occur_item_previous * 2, code, 1)
	# end for
	
	# check last character
	# -s, remove
	code = re.sub (r's$', r'', code)
	# -ay, replace by -y
	code = re.sub (r'ay$', r'y', code)
	# -a, remove
	code = re.sub (r'a$', r'', code)
	
	# return nysiis code
	return code

def caverphone (term):
	"returns the language key using the caverphone algorithm 2.0"

	# Developed at the University of Otago, New Zealand.
	# Project: Caversham Project (http://caversham.otago.ac.nz)
	# Developer: David Hood, University of Otago, New Zealand
	# Contact: caversham@otago.ac.nz
	# Project Technical Paper: http://caversham.otago.ac.nz/files/working/ctp150804.pdf
	# Version 2.0 (2004-08-15)

	code = ""

	i = 0
	term_length = len(term)

	if (term_length == 0):
		# empty string ?
		return code
	# end if

	# convert to lowercase
	code = string.lower(term)

	# remove anything not in the standard alphabet (a-z)
	code = re.sub(r'[^a-z]', '', code)

	# remove final e
	if code.endswith("e"):
		code = code[:-1]

	# if the name starts with cough, rough, tough, enough or trough -> cou2f (rou2f, tou2f, enou2f, trough)
	code = re.sub(r'^([crt]|(en)|(tr))ough', r'\1ou2f', code)

	# if the name starts with gn -> 2n
	code = re.sub(r'^gn', r'2n', code)

	# if the name ends with mb -> m2
	code = re.sub(r'mb$', r'm2', code)

	# replace cq -> 2q
	code = re.sub(r'cq', r'2q', code)
	
	# replace c[i,e,y] -> s[i,e,y]
	code = re.sub(r'c([iey])', r's\1', code)
	
	# replace tch -> 2ch
	code = re.sub(r'tch', r'2ch', code)
	
	# replace c,q,x -> k
	code = re.sub(r'[cqx]', r'k', code)
	
	# replace v -> f
	code = re.sub(r'v', r'f', code)
	
	# replace dg -> 2g
	code = re.sub(r'dg', r'2g', code)
	
	# replace ti[o,a] -> si[o,a]
	code = re.sub(r'ti([oa])', r'si\1', code)
	
	# replace d -> t
	code = re.sub(r'd', r't', code)
	
	# replace ph -> fh
	code = re.sub(r'ph', r'fh', code)

	# replace b -> p
	code = re.sub(r'b', r'p', code)
	
	# replace sh -> s2
	code = re.sub(r'sh', r's2', code)
	
	# replace z -> s
	code = re.sub(r'z', r's', code)

	# replace initial vowel [aeiou] -> A
	code = re.sub(r'^[aeiou]', r'A', code)

	# replace all other vowels [aeiou] -> 3
	code = re.sub(r'[aeiou]', r'3', code)

	# replace j -> y
	code = re.sub(r'j', r'y', code)

	# replace an initial y3 -> Y3
	code = re.sub(r'^y3', r'Y3', code)
	
	# replace an initial y -> A
	code = re.sub(r'^y', r'A', code)

	# replace y -> 3
	code = re.sub(r'y', r'3', code)
	
	# replace 3gh3 -> 3kh3
	code = re.sub(r'3gh3', r'3kh3', code)
	
	# replace gh -> 22
	code = re.sub(r'gh', r'22', code)

	# replace g -> k
	code = re.sub(r'g', r'k', code)

	# replace groups of s,t,p,k,f,m,n by its single, upper-case equivalent
	for single_letter in ["s", "t", "p", "k", "f", "m", "n"]:
		otherParts = re.split(single_letter + "+", code)
		code = string.join(otherParts, string.upper(single_letter))
	
	# replace w[3,h3] by W[3,h3]
	code = re.sub(r'w(h?3)', r'W\1', code)

	# replace final w with 3
	code = re.sub(r'w$', r'3', code)

	# replace w -> 2
	code = re.sub(r'w', r'2', code)

	# replace h at the beginning with an A
	code = re.sub(r'^h', r'A', code)

	# replace all other occurrences of h with a 2
	code = re.sub(r'h', r'2', code)

	# replace r3 with R3
	code = re.sub(r'r3', r'R3', code)

	# replace final r -> 3
	code = re.sub(r'r$', r'3', code)

	# replace r with 2
	code = re.sub(r'r', r'2', code)

	# replace l3 with L3
	code = re.sub(r'l3', r'L3', code)
	
	# replace final l -> 3
	code = re.sub(r'l$', r'3', code)
	
	# replace l with 2
	code = re.sub(r'l', r'2', code)

	# remove all 2's
	code = re.sub(r'2', r'', code)

	# replace the final 3 -> A
	code = re.sub(r'3$', r'A', code)
	
	# remove all 3's
	code = re.sub(r'3', r'', code)

	# extend the code by 10 '1' (one)
	code += '1' * 10
	
	# take the first 10 characters
	caverphoneCode = code[:10]
	
	# return caverphone code
	return caverphoneCode

