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

public:
	BenTexture(TextureMapping3D *map_) : map(map_) {

	}

	virtual Spectrum Evaluate(const DifferentialGeometry &) const;

	virtual ~BenTexture() {
		delete map;
	}
};

Texture<float> *CreateBenFloatTexture(const Transform &tex2world, const TextureParams &tp);
BenTexture *CreateBenSpectrumTexture(const Transform &tex2world, const TextureParams &tp);

#endif