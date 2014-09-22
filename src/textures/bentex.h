/*
 * Ben's Texture Thingy
 */

#ifndef PBRT_BENTEX_H
#define PBRT_BENTEX_H

#include "pbrt.h"
#include "texture.h"
#include "paramset.h"

class BenTexture : public Texture<Spectrum> {
private:
	TextureMapping3D *map;
	// noise applied to imput of main grain function
	float wobble0, wobble0_freq;
	// noise applied to input y-values to give varying grain spacing
	float wobble1, wobble1_freq;
	// control over the outer waveform applied to wobble1
	float wobble2, wobble2_freq;
	// low-frequency primary grain
	float grain0_freq;
	// higher-frequency secondary grain
	float grain1_freq;
	// color interpolation noise - grain-plane frequency multiplier, primary frequency
	float noise_stretch, noise_freq, noise_weight;
	// lerp between primary colors
	float blotch_freq, blotch_weight;
	// two primary colors and one (dark) ring color
	Spectrum color0, color1, color2;

public:
	BenTexture(TextureMapping3D *map_, const TextureParams &tp) : map(map_) {
		// defaults are set for the stanford dragon
		wobble0_freq = tp.FindFloat("wobble0_freq", 1);
		wobble0 = tp.FindFloat("wobble0", 0.08);
		wobble1_freq = tp.FindFloat("wobble1_freq", 1);
		wobble1 = tp.FindFloat("wobble1", 0.3);
		wobble2_freq = tp.FindFloat("wobble2_freq", 1);
		wobble2 = tp.FindFloat("wobble2", 0.08 / wobble2_freq);
		grain0_freq = tp.FindFloat("grain0_freq", 40);
		grain1_freq = tp.FindFloat("grain1_freq", 200);
		noise_stretch = tp.FindFloat("noise_stretch", 20);
		noise_freq = tp.FindFloat("noise_freq", 5);
		noise_weight = tp.FindFloat("noise_weight", 0.2);
		blotch_freq = tp.FindFloat("blotch_freq", 5);
		blotch_weight = tp.FindFloat("blotch_weight", 0.3);
		float c0[3] { 192.0 / 255.0, 134.0 / 255.0, 84.0 / 255.0 };
		float c1[3] { 211.0 / 255.0, 102.0 / 255.0, 32.0 / 255.0 };
		float c2[3] { 85.0 / 255.0, 53.0 / 255.0, 22.0 / 255.0 };
		color0 = tp.FindSpectrum("color0", Spectrum::FromRGB(c0));
		color1 = tp.FindSpectrum("color1", Spectrum::FromRGB(c1));
		color2 = tp.FindSpectrum("color2", Spectrum::FromRGB(c2));
	}

	virtual Spectrum Evaluate(const DifferentialGeometry &) const;

	virtual ~BenTexture() {
		delete map;
	}
};

Texture<float> *CreateBenFloatTexture(const Transform &tex2world, const TextureParams &tp);
BenTexture *CreateBenSpectrumTexture(const Transform &tex2world, const TextureParams &tp);

#endif