from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from view.mesh_plotter.mesh_plots import plot_dataset, plot_mesh, save_dataset_plot
from mesh_model.reader import read_dataset, read_gmsh, read_json, read_medit
from environment.actions.smoothing import smoothing_mean

# dataset = read_dataset("../../training/dataset/results/new-score-v2")
# plot_dataset(dataset)
# save_dataset_plot(dataset, "results_5_darts_new.png")
# print("File saved")

cmap = read_json("../../mesh_files/mesh_to_cleanup/mesh_102.json") #read_gmsh("../../mesh_files/imr3.msh")
# ma = QuadMeshTopoAnalysis(cmap)
# smoothing_mean(cmap)
plot_mesh(cmap, scores=True)
