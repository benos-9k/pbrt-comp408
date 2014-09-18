#include "bentex.h"

Spectrum BenTexture::Evaluate(const DifferentialGeometry &dg) const {
	
	float f[3] { 0.1, 0.2, 0.9};
	Spectrum s = Spectrum::FromRGB(f);
	return s;
}

Texture<float> * CreateBenFloatTexture(const Transform &tex2world, const TextureParams &tp) {
	return nullptr;
}

BenTexture * CreateBenSpectrumTexture(const Transform &tex2world, const TextureParams &tp) {
	TextureMapping3D *map = new IdentityMapping3D(tex2world);
	return new BenTexture(map);
}