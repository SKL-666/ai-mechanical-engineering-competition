import unittest
import numpy as np
import torch
from PIL import Image
from earthmoving.round2_data import frame_ids,resize_crop,Round2Dataset,D2
from earthmoving.round2_model import ActionNet
from earthmoving.round2_metrics import segment_counts,edit_similarity,temporal_metrics,boundary_stats
from scripts.round2_continuous import render_predictions
from scripts.round2_action import guard_truth


class Round2Contracts(unittest.TestCase):
    def test_interval_sampling_does_not_scale_with_duration(self):
        data=Round2Dataset.__new__(Round2Dataset)
        data.training=True
        data.cfg={'seed':42,'samples_per_epoch':4000,'sampling':'intervals','training_mode':'core'}
        data.rows=[{'video_id':'a','raw_line':1,'label':0,'core_start':10,'core_end':12},
                   {'video_id':'b','raw_line':2,'label':1,'core_start':100,'core_end':10100}]
        data.groups=[[0],[1]]
        data.set_epoch(1)
        counts=[sum(r['label']==c for r,_ in data.schedule) for c in [0,1]]
        self.assertTrue(all(1800<n<2200 for n in counts),counts)
        for r,_ in data.schedule:
            self.assertTrue(r['core_start']<=r['start_frame']<r['end_frame']<=r['core_end'])
        first=[r for r,_ in data.schedule]
        data.set_epoch(1)
        self.assertEqual(first,[r for r,_ in data.schedule])

    def test_latest_frame_and_short_repeat(self):
        self.assertEqual(frame_ids({'start_frame':7,'end_frame':71},1).tolist(),[70])
        self.assertEqual(frame_ids({'start_frame':7,'end_frame':8},8).tolist(),[7]*8)

    def test_causal_model_cannot_use_future_features(self):
        torch.manual_seed(3);model=ActionNet(True).eval();x=torch.rand(2,3,8,32,32)
        y=model.timeline(x)
        changed=x.clone();changed[:,:,5:]=torch.rand_like(changed[:,:,5:])
        torch.testing.assert_close(y[:,:,:5],model.timeline(changed)[:,:,:5])
        self.assertGreater((y[:,:,-1]-model.timeline(x.flip(2))[:,:,-1]).abs().max().item(),1e-7)

    def test_letterbox_preserves_content_ratio(self):
        im=resize_crop(Image.new('RGB',(100,50),(255,0,0)),100,True)
        a=np.array(im);self.assertTrue((a[25:75,:,0]==255).all());self.assertTrue((a[:25]==128).all())

    def test_temporal_scores_perfect_and_oversegmented(self):
        truth=np.array([0,0,-1,1,1,2,2])
        perfect=np.array([0,0,0,1,1,2,2])
        m=temporal_metrics(truth,perfect)
        self.assertEqual(m['segment_f1']['0.5'],1);self.assertEqual(m['edit'],1)
        self.assertEqual(segment_counts([0,0,0,0],[0,1,0,1],.1),(1,3,0))
        self.assertAlmostEqual(edit_similarity([0,1],[0,2]),.5)

    def test_boundary_uncertainty_is_interval(self):
        m=boundary_stats(np.array([0,0,-1,-1,1,1]),np.array([0,0,0,1,1,1]))
        self.assertEqual(m['matched'],1);self.assertEqual(m['matched_distance_to_truth_bracket_mean_frames'],0)

    def test_prediction_generation_does_not_need_truth(self):
        meta={'frame_count':90,'video_id':'v','equipment_ids':['1'],'valid_tracks':{'1':list(range(90))}}
        rows=[{'equipment_id':'1','available_at_frame':63,'probabilities':[1,0,0,0,0]},
              {'equipment_id':'1','available_at_frame':71,'probabilities':[0,1,0,0,0]}]
        seg,dense=render_predictions(meta,rows,8,1)
        self.assertEqual(seg[0]['start_frame'],63);self.assertEqual(dense['1'][71],1)
        for path in ['derived/train_continuous.jsonl','derived/test_clips.jsonl','Data/Labels/1/excavator.txt']:
            with self.assertRaises(RuntimeError):guard_truth('open',(path,'r'))


if __name__=='__main__':unittest.main()
