#include "bentex.h"

Spectrum BenTexture::Evaluate(const DifferentialGeometry &dg) const {
	
	Point p;
	Vector dpdx, dpdy;
	p = map->Map(dg, &dpdx, &dpdy);

	// the min value of FBm() is less than -1. wtf?

	// sawtooth-ish function (smooth)
	auto thing = [&](float x) {
		return pow(pow(cos(M_PI * x), 2), 0.1 + pow(fmod(fmod(x, 1) + 1, 1), 3) * 30);
	};

	Point p2 = p;
	p2.x *= 0.08;
	p2.z *= 0.08;
	p2.y += 0.08 * pow(sin(2.0 * M_PI * p2.y + 0.3 * FBm(p, dpdx, dpdy, 1, 2)), 2.0);
	//p2.y += 0.08 * cos(p2.z * 50.0);
	float n = FBm(p2, dpdx, dpdy, 1.0, 4);
	float m = 0.08 * n;

	// eval points for wood 'waves'
	float x0 = 40 * (p.y + m);
	float x1 = 200 * (p.y + m);

	// interpolation noise
	Point p3 = p;
	p3.y *= 5;
	p3.x *= 100;
	p3.z *= 100;
	float b0 = 0.2 * (FBm(p3, dpdx, dpdy, 1.0, 8) + n);

	// light <-> dark interpolant
	float a0 = (0.6 + b0) * thing(x0) + (0.4 - b0) * thing(x1);
	

	float c0[3] { 192.0 / 255.0, 134.0 / 255.0, 84.0 / 255.0 };
	float c1[3] { 211.0 / 255.0, 102.0 / 255.0, 32.0 / 255.0 };
	float c2[3] { 85.0 / 255.0, 53.0 / 255.0, 22.0 / 255.0 };

	Spectrum s0 = Spectrum::FromRGB(c0);
	Spectrum s1 = Spectrum::FromRGB(c1);
	Spectrum s2 = Spectrum::FromRGB(c2);
	
	// colour blotchiness noise
	float b2 = 0.3 * FBm(p * 5, dpdx, dpdy, 0.5, 8);

	return ((0.6 + b2) * s0 + (0.4 - b2) * s1) * (1.0 - a0) + s2 * a0;
}

Texture<float> * CreateBenFloatTexture(const Transform &tex2world, const TextureParams &tp) {
	return nullptr;
}

BenTexture * CreateBenSpectrumTexture(const Transform &tex2world, const TextureParams &tp) {
	TextureMapping3D *map = new IdentityMapping3D(tex2world);
	return new BenTexture(map);
}