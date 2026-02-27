import unittest
import os
import copy

from stable_baselines3 import PPO
from training.exploit_SB3_policy import testPolicy

from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from environment.actions.quadrangular_actions import flip_edge_cw_ids, split_edge_ids, collapse_edge_ids
from mesh_model.mesh_struct.mesh import Mesh
from view.mesh_plotter.mesh_plots import plot_mesh
from mesh_model.reader import read_gmsh

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestQuadActions(unittest.TestCase):

    def test_IMR_example(self):
        cmap = read_gmsh(os.path.join(TESTFILE_FOLDER, 'IMR_example_mesh.msh'))
        ma = QuadMeshTopoAnalysis(cmap)
        plot_mesh(cmap, scores=True, irregularities=True)

        nodes_score, mesh_score, mesh_ideal_score = ma.global_score()
        self.assertEqual((16, 0), (mesh_score, mesh_ideal_score))

        flip_edge_cw_ids(ma,16,19)
        flip_edge_cw_ids(ma, 5, 17)
        flip_edge_cw_ids(ma, 22, 10)
        collapse_edge_ids(ma, 20, 31)
        collapse_edge_ids(ma, 32, 23)
        split_edge_ids(ma, 25,14)
        split_edge_ids(ma, 27, 18)
        flip_edge_cw_ids(ma, 21, 3)
        nodes_score, mesh_score, mesh_ideal_score = ma.global_score()
        self.assertEqual((30, 0), (mesh_score, mesh_ideal_score))
        plot_mesh(cmap, scores=True, irregularities=True)


    def test_toy_example_with_cleanup_ppt(self):
        # Test to cleanup one face on boundary
        cmap = Mesh()

        n00 = cmap.add_node(0, 0, 0)
        n02 = cmap.add_node(0, 2, 0)
        n20 = cmap.add_node(2, 0, 0)
        n22 = cmap.add_node(2, 2, 0)

        n21 = cmap.add_node(2, 1, 1)
        n01 = cmap.add_node(0,1,1)
        n12 = cmap.add_node(1, 2, 1)
        n10 = cmap.add_node(1,0,1)

        n11 = cmap.add_node(1,1,2)
        ni1 = cmap.add_node(0.75,0.75,2)
        ni2 = cmap.add_node(1.25,1.25,2)

        q1 = cmap.add_quad(n00, n10, ni1, n01)
        q2 = cmap.add_quad(n02, n01, ni1, n11)
        q3 = cmap.add_quad(n22, n12, ni2, n21)
        q4 = cmap.add_quad(n20, n21, ni2, n11)
        q5 = cmap.add_quad(n10, n20, n11, ni1)
        q6 = cmap.add_quad(n02, n11, ni2, n12)
        cmap.set_twin_pointers()
        ma = QuadMeshTopoAnalysis(cmap)

        # Score initial
        nodes_score_init, mesh_score_init, mesh_ideal_score = ma.global_score()
        print(f"Score initial : {mesh_score_init}, Score idéal : {mesh_ideal_score}")
        plot_mesh(cmap, debug=True, scores=True, irregularities=True)

        # Configuration directe pour plus de fiabilité
        config = {
            "eval": {
                "eval_env_id": "Quadmesh-v0",
                "max_episode_steps": 300,
                "n_darts_selected": 5,
                "deep": 36,
                "obs_size": 180,
                "render_mode": None,  # "human"
                "action_restriction": False,
                "with_quality_observation": False,
            },
            "env": {
                "analysis_type": "boundary",
            }
        }

        # Charger le modèle
        model_path = os.path.join(os.path.dirname(__file__), '../training/policy_saved/e5/cleanup-flag-newscore-v2/cleanup-flag-newscore-v2_4000000_steps.zip')
        if os.path.exists(model_path):
            model = PPO.load(model_path)

            # Créer un dataset avec un seul maillage
            dataset = [copy.deepcopy(cmap)]

            # Utiliser testPolicy pour optimiser le maillage
            df_results = testPolicy(model, n_eval_episodes=1, config=config, dataset=dataset)

            # Récupérer le maillage final optimisé
            optimized_mesh = df_results.loc[0, "final_mesh"]

            # Vérifier le score final
            ma_final = QuadMeshTopoAnalysis(optimized_mesh)
            nodes_score_final, mesh_score_final, _ = ma_final.global_score()
            print(f"Score final : {mesh_score_final}, Score idéal : {mesh_ideal_score}")

            plot_mesh(optimized_mesh, debug=True, scores=True, irregularities=True)

            # Assertion : vérifier que le score final est 0
            self.assertEqual(mesh_ideal_score, mesh_score_final,
                           f"Le score final devrait être {mesh_ideal_score}, mais obtenu {mesh_score_final}")
        else:
            print(f"Modèle non trouvé à : {model_path}")

    def test_toy_example__without_cleanup_ppt(self):
        # Test to cleanup one face on boundary
        cmap = Mesh()

        n00 = cmap.add_node(0, 0, 0)
        n02 = cmap.add_node(0, 2, 0)
        n20 = cmap.add_node(2, 0, 0)
        n22 = cmap.add_node(2, 2, 0)

        n21 = cmap.add_node(2, 1, 1)
        n01 = cmap.add_node(0,1,1)
        n12 = cmap.add_node(1, 2, 1)
        n10 = cmap.add_node(1,0,1)

        n11 = cmap.add_node(1,1,2)
        ni1 = cmap.add_node(0.75,0.75,2)
        ni2 = cmap.add_node(1.25,1.25,2)

        q1 = cmap.add_quad(n00, n10, ni1, n01)
        q2 = cmap.add_quad(n02, n01, ni1, n11)
        q3 = cmap.add_quad(n22, n12, ni2, n21)
        q4 = cmap.add_quad(n20, n21, ni2, n11)
        q5 = cmap.add_quad(n10, n20, n11, ni1)
        q6 = cmap.add_quad(n02, n11, ni2, n12)
        cmap.set_twin_pointers()
        ma = QuadMeshTopoAnalysis(cmap)

        # Score initial
        nodes_score_init, mesh_score_init, mesh_ideal_score = ma.global_score()
        print(f"Score initial : {mesh_score_init}, Score idéal : {mesh_ideal_score}")
        plot_mesh(cmap, debug=True, scores=True, irregularities=True)

        # Configuration directe pour plus de fiabilité
        config = {
            "eval": {
                "eval_env_id": "Quadmesh-v0",
                "max_episode_steps": 300,
                "n_darts_selected": 5,
                "deep": 36,
                "obs_size": 180,
                "render_mode": None,  # "human"
                "action_restriction": False,
                "with_quality_observation": False,
            },
            "env": {
                "analysis_type": "topo",
            }
        }

        # Charger le modèle
        model_path = os.path.join(os.path.dirname(__file__), '../training/policy_saved/e1/full-dataset-obs36-5darts-v0/full-dataset-obs36-5darts-v0_5500000_steps.zip')
        if os.path.exists(model_path):
            model = PPO.load(model_path)

            # Créer un dataset avec un seul maillage
            dataset = [copy.deepcopy(cmap)]

            # Utiliser testPolicy pour optimiser le maillage
            df_results = testPolicy(model, n_eval_episodes=5, config=config, dataset=dataset)

            # Récupérer le maillage final optimisé
            optimized_mesh = df_results.loc[0, "final_mesh"]

            # Vérifier le score final
            ma_final = QuadMeshTopoAnalysis(optimized_mesh)
            nodes_score_final, mesh_score_final, _ = ma_final.global_score()
            print(f"Score final : {mesh_score_final}, Score idéal : {mesh_ideal_score}")

            plot_mesh(optimized_mesh, scores=True, irregularities=True)

            # Assertion : vérifier que le score final est 0
            self.assertEqual(mesh_ideal_score, mesh_score_final,
                           f"Le score final devrait être {mesh_ideal_score}, mais obtenu {mesh_score_final}")
        else:
            print(f"Modèle non trouvé à : {model_path}")
