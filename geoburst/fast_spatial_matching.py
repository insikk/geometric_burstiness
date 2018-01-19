# Fast Spatial Matching
import _fast_spatial_matching
from .geometric_transforms import Transformation, AffineFeatureMatch, FeatureGeometryAffine



class AffineMatch:
    def __init__(self, query_feature_index, query_keypoint):
        self.feature1 = FeatureGeometryAffine()
        self.feature1.feature_id_ = query_feature_index
        self.feature1.setPosition(query_keypoint[0], query_keypoint[1])
        self.feature1.a_ = query_keypoint[2]
        # TODO: check if 2b is correct or b is correct.

        # Input Keypoint: u,v,a,b,c    in    a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1
        #     with (0,0) at image top left corner
        # GeoBurst compatible Keypoint:
        #     a * x^2 + b * xy + c * y^2 = 1 describes all points on the sphere
        self.feature1.b_ = query_keypoint[3]
        self.feature1.c_ = query_keypoint[4]

        self.features2 = []

        # For SFM, below var does not affet the result
        self.word_ids = []

    def get_object(self):
        return AffineFeatureMatch(self.feature1, self.features2, self.word_ids)


class FastSpatialMatching ():
    def __init__(self):
        self._impl = _fast_spatial_matching.PyFastSpatialMatching()

    def perform_spatial_verification(self, matches, kp1, kp2):
        """
        get opencv style match and keypoints, run match
        kp is x,y,a,b,c for each row
        """
        affine_matches = []
        # TODO: handle multi match.
        # when one key points matches multiple key points. check data structure for this and fsm.

        for match_idx, m in enumerate(matches):
            am = AffineMatch(m.queryIdx, kp1[m.queryIdx])

            feature2 = FeatureGeometryAffine()
            feature2.feature_id_ = m.trainIdx
            feature2.setPosition(kp2[m.trainIdx][0], kp2[m.trainIdx][1])
            feature2.a_ = kp2[m.trainIdx][2]
            feature2.b_ = kp2[m.trainIdx][3]
            feature2.c_ = kp2[m.trainIdx][4]

            am.features2.append(feature2)
            am.word_ids.append(m.trainIdx) # pass dummy word_id. We may don't use value for FSM.

            affine_matches.append(am)

        # TODO: check do we need sort by size of features2 (smaller first) for matches. for FSM
        # print("num matches:", len(affine_matches))
        match_list = affine_matches
        match_obj_list = []
        for m in affine_matches:
            match_obj_list.append(m.get_object())
        
        best_num_inliers, transform, inliers = self._impl.PerformSpatialVerification(match_obj_list)
        return transform, inliers
