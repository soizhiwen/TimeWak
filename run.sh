#!/bin/bash

CONFIGS=("stocks" "etth" "mujoco" "energy" "fmri")
METHODS=("tr" "gs" "htw" "tabwak" "tabwak-t" "timewak" "spatddim" "spatbdia" "tempddim")
WINDOW_SIZES=(24 64 128)
ATTACKS=("offset" "crop" "insert")

# Attacks configuration
OFFSET_FACTORS=(0.05 0.3)
CROP_FACTORS=(0.05 0.3)
INSERT_FACTORS=(0.05 0.3)

declare -A METHOD_NAMES
METHOD_NAMES["tr"]="TR"
METHOD_NAMES["gs"]="GS"
METHOD_NAMES["htw"]="HTW"
METHOD_NAMES["tabwak"]="TabWak"
METHOD_NAMES["tabwak-t"]="TabWakT"
METHOD_NAMES["timewak"]="TimeWak"
METHOD_NAMES["spatddim"]="SpatDDIM"
METHOD_NAMES["spatbdia"]="SpatBDIA"
METHOD_NAMES["tempddim"]="TempDDIM"

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
	for CONFIG in "${CONFIGS[@]}"; do

		# Training (Uncomment below to train a new model)
		# python main.py --name ${CONFIG}-train-${WINDOW_SIZE} \
		# 	--config_file ./Config/${CONFIG}.yaml --gpu 0 \
		# 	--train --window_size $WINDOW_SIZE

		# Sampling without watermark
		python main.py --name ${CONFIG}-sample-${WINDOW_SIZE} \
			--config_file ./Config/${CONFIG}.yaml --gpu 0 \
			--sample 0 --milestone 10 --window_size $WINDOW_SIZE

		for METHOD in "${METHODS[@]}"; do

			if [[ "$METHOD" == "htw" ]]; then
				# HTW: Post-watermark
				python main.py --name ${CONFIG}-sample-${METHOD}-${WINDOW_SIZE} \
					--config_file ./Config/${CONFIG}.yaml --gpu 0 --sample 0 \
					--watermark ${METHOD_NAMES[$METHOD]} --window_size $WINDOW_SIZE \
					--post_wm ./OUTPUT/${CONFIG}-sample-${WINDOW_SIZE}/ddpm_fake_${CONFIG}-sample-${WINDOW_SIZE}.npy

				# HTW: Detection with watermark
				python main.py --name ${CONFIG}-detect-${METHOD}-${WINDOW_SIZE} \
					--config_file ./Config/${CONFIG}.yaml --gpu 0 \
					--detect ./OUTPUT/${CONFIG}-sample-${METHOD}-${WINDOW_SIZE}/ddpm_fake_${CONFIG}-sample-${METHOD}-${WINDOW_SIZE}.npy \
					--watermark ${METHOD_NAMES[$METHOD]} --window_size $WINDOW_SIZE

			else
				# Sampling with watermark
				python main.py --name ${CONFIG}-sample-${METHOD}-${WINDOW_SIZE} \
					--config_file ./Config/${CONFIG}.yaml --gpu 0 --sample 0 --milestone 10 \
					--watermark ${METHOD_NAMES[$METHOD]} --window_size $WINDOW_SIZE

				# Detection without watermark
				python main.py --name ${CONFIG}-detect-wo-${METHOD}-${WINDOW_SIZE} \
					--config_file ./Config/${CONFIG}.yaml --gpu 0 \
					--detect ./OUTPUT/${CONFIG}-sample-${WINDOW_SIZE}/ddpm_fake_${CONFIG}-sample-${WINDOW_SIZE}.npy \
					--milestone 10 --watermark ${METHOD_NAMES[$METHOD]} --window_size $WINDOW_SIZE

				# Detection with watermark
				python main.py --name ${CONFIG}-detect-${METHOD}-${WINDOW_SIZE} \
					--config_file ./Config/${CONFIG}.yaml --gpu 0 \
					--detect ./OUTPUT/${CONFIG}-sample-${METHOD}-${WINDOW_SIZE}/ddpm_fake_${CONFIG}-sample-${METHOD}-${WINDOW_SIZE}.npy \
					--milestone 10 --watermark ${METHOD_NAMES[$METHOD]} --window_size $WINDOW_SIZE

			fi

			for ATTACK in "${ATTACKS[@]}"; do

				# Select the appropriate attack factors
				if [[ "$ATTACK" == "offset" ]]; then
					ATTACK_FACTORS=("${OFFSET_FACTORS[@]}")

				elif [[ "$ATTACK" == "crop" ]]; then
					ATTACK_FACTORS=("${CROP_FACTORS[@]}")

				elif [[ "$ATTACK" == "insert" ]]; then
					ATTACK_FACTORS=("${INSERT_FACTORS[@]}")

				fi

				for ATTACK_FACTOR in "${ATTACK_FACTORS[@]}"; do

					# Attack with watermark
					python main.py --name ${CONFIG}-attack-${METHOD}-${ATTACK}-${ATTACK_FACTOR}-${WINDOW_SIZE} \
						--config_file ./Config/${CONFIG}.yaml --gpu 0 \
						--detect ./OUTPUT/${CONFIG}-sample-${METHOD}-${WINDOW_SIZE}/ddpm_fake_${CONFIG}-sample-${METHOD}-${WINDOW_SIZE}.npy \
						--milestone 10 --watermark ${METHOD_NAMES[$METHOD]} --window_size $WINDOW_SIZE \
						--attack ${ATTACK} --attack_factor ${ATTACK_FACTOR}

				done
			done
		done
	done
done
