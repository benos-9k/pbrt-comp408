
#ifndef PBRT_ACCELERATORS_OCTREE_H
#define PBRT_ACCELERATORS_OCTREE_H

#include "pbrt.h"
#include "primitive.h"

class OctreeAccel : public Aggregate {
private:
	class Node;

	Node *root = nullptr;

public:
	OctreeAccel(const vector<Reference<Primitive>> &prims);
	~OctreeAccel();

	BBox WorldBound() const;
	bool CanIntersect() const { return true; }

	bool Intersect(const Ray &ray, Intersection *isect) const;
	bool IntersectP(const Ray &ray) const;
};

inline OctreeAccel *CreateOctreeAccelerator(const vector<Reference<Primitive>> &prims, const ParamSet &ps) {
	return new OctreeAccel(prims);
}

#endif