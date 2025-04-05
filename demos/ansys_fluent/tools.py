"""
Tool Usage Example.

This script shows how function-based tools can be made accessible to LLMs.
"""

# pylint: disable=import-error
# pylint: disable=wrong-import-order
# pylint: disable=unsubscriptable-object

# Prepare imports.
import os
import re
import glob
import shutil
import cadquery as cq
import ansys.fluent.core as pyf


class Tools:
    """
    This class defines functions that are used as tools by LLMs, as well as their tool schemas
    describing how the tools are used.
    """

    # Prepare tool schemas.
    # WHY: The models need to know what tools they have and how they tools work.
    TOOL_SCHEMA = [{
        "name": "evaluate_design",
        "description": (
            "Given coordinates of a middle spline control point and an end spline control point "
            "which represent the curve that consittutes the wall of a diffusor. The 2D coordinats "
            "are converted to splines and converted into a 3D diffusor. The function returns "
            "performance metrics like static pressure rise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "control_point_1": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "description": ("Coorddinates of middle spline control point in format "
                                        "[x, r], where x is the distance from the diffusor inlet "
                                        "and r is the distance from the centeral line of rotation. "
                                        "Example [10, 1.5]")
                    },
                    "minItems": 2,
                    "maxItems": 2,
                },
                "control_point_2": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "description": ("Coorddinates of middle spline control point in format "
                                        "[x, r], where x is the distance from the diffusor inlet "
                                        "and r is the distance from the centeral line of rotation. "
                                        "Example [10, 1.5]")
                    },
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            "required": ["control_point_1", "control_point_2"]
        }
    }]

    def __init__(self):

        # Initialising session details.
        # WHY: Used later as globals. Keeping pylint happy.
        self.sesh = {}
        self.solver = {}

        # Define constants.
        # WHY: Simplifies making changes later.
        self.iterations = 750
        self.inlet_point = [0, 1]
        self.boundary_planes = {"body": "freeparts-open-cascade-step-translator-7.7-1:1",
                                "inlet": "freeparts-open-cascade-step-translator-7.7-1:1:11",
                                "outlet": "freeparts-open-cascade-step-translator-7.7-1:1:12"}

        # Define and ensure base dicrectory exists.
        # WHY: Avoids future read/write errors.
        self.base_dir = os.path.expanduser("~/Downloads/HyTE/demos/ansys_fluent/")
        self.output_dir = os.path.expanduser("~/Downloads/HyTE/demos/ansys_fluent//Outputs")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_design(self, control_point_1: tuple[float, float],
                        control_point_2: tuple[float, float]) -> dict:
        """
        Generate a diffusor based on spline control points, CFD it, and return relevant metrics.

        Parameters
        ----------
        control_point_1 : x, r (horizontal distance, radius) position of mid diffusor spline point.
        control_point_2 : x, r (horizontal distance, radius) position of end diffusor spline point.

        Returns
        -------
        dict : A key-value store of metrics. e.g. static pressure rise.
        """

        # Create geometry.
        # WHY: Geometry is created, ready for meshing.
        self._create_geometry(control_point_1, control_point_2)

        # Mesh geometry.
        # WHY: Mesh is used to run CFD of design.
        self._mesh_geometry()

        # Run solver.
        # WHY: Converged solution is used to calculate performance metrics.
        self._run_cfd()

        # Post-process metrics.
        # WHY: Extracts and converts perf. metrics into a neat dictionary format that AI can read.
        return {
            "control_point_1": control_point_1,
            "control_point_2": control_point_2,
            "results": self._post_process(),
        }

    def cleanup(self, pattern: str = "FM_*") -> None:
        """
        Deletes files matching pattern, mostly used to remove garbage fluent leaves behind.

        Parameters
        ----------
        pattern : Pattern to check against when deleting files / folders.

        Returns
        -------
        None
        """

        # Construct full pattern path and iteratively delete files / folders inside pattern.
        full_pattern = os.path.join(self.base_dir, pattern)
        for path in glob.glob(full_pattern):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    ## SUPPORT FUNCTIONS ##

    def _create_geometry(self, control_point_1: tuple[float, float],
                         control_point_2: tuple[float, float]) -> None:
        """
        Generates diffusor geometery via spines of control points and saves it to a step file.

        Parameters
        ----------
        control_point_1 : x, r (horizontal distance, radius) position of mid diffusor spline point.
        control_point_2 : x, r (horizontal distance, radius) position of end diffusor spline point.

        Returns
        -------
        None
        """

        # Check whether control points are reasonable.
        # WHY: Avoids giving terrible geometry to Ansys for meshing.
        if not self.inlet_point[0] <= control_point_1[0] <= control_point_2[0]:
            raise ValueError("X coordinate of middle control point must be < x of control point 2.")

        # Create diffusor solid block based on sizes.
        # WHY: Diffusor will be read by Ansys mesher (whatever its called).
        diffusor = (
            cq.Workplane("XY")
            .circle(self.inlet_point[1])
            .workplane(offset=control_point_1[0])
            .circle(control_point_1[1])
            .workplane(offset=control_point_2[0] - control_point_1[0])
            .circle(control_point_2[1])
            .loft(combine=True)
        )

        # Export the hollow cylinder as an STL with specified tolerance.
        # WHY: Saving STL so it can be consumed by Fluent later.
        cq.exporters.export(diffusor, f"{self.output_dir}/design.step", exportType='STEP')

        # Create a fluent sesson.
        # WHY: So we can interact with the mesher and solver.
        self.sesh = pyf.launch_fluent(precision="double", processor_count=2,
                                      mode="meshing", ui_mode='gui')

    def _mesh_geometry(self) -> None:
        """
        Converts CAD file into mesh that can be processed by solver.

        Returns
        -------
        None
        """

        # Create meshing workflow.
        # WHY: Allows for controlling Fluent's meshing.
        workflow = self.sesh.workflow
        workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
        wf_to = workflow.TaskObject

        # Import geometry and generate surface mesh.
        # WHY: Preparing surface geometry to get volume geometry later.
        wf_to['Import Geometry'].Arguments.set_state({
            r'FileName': f"{self.output_dir}/design.step",
            r'ImportCadPreferences': {r'MaxFacetLength': 0,}, r'LengthUnit': r'mm'})
        wf_to['Import Geometry'].Execute()
        wf_to['Add Local Sizing'].AddChildAndUpdate(DeferUpdate=False)
        wf_to['Generate the Surface Mesh'].Arguments.set_state({r'SeparationRequired': r'Yes'})
        wf_to['Generate the Surface Mesh'].Execute()

        # Define inlet and outlets faces by angle.
        # WHY: Inlets and outlets will be used for boundary conditions. Using angles for simplicty.
        wf_to['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=False)
        wf_to['Describe Geometry'].Arguments.set_state({
            r'NonConformal': r'No',
            r'SetupType': r'The geometry consists of only fluid regions with no voids'})
        wf_to['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=True)
        wf_to['Describe Geometry'].Arguments.set_state({
            r'NonConformal': r'No',
            r'SetupType': r'The geometry consists of only fluid regions with no voids',
            r'WallToInternal': r'Yes'})
        wf_to['Describe Geometry'].Execute()

        # Define boundry condition planes, some of their flow properties and create the volume mesh.
        # WHY: This "complete" mesh will be consumed by the solver, which will run CFD of it.
        wf_to['Update Boundaries'].Arguments.set_state({
            r'BoundaryZoneList': [f'{self.boundary_planes["outlet"]}',
                                  f'{self.boundary_planes["inlet"]}'],
            r'BoundaryZoneTypeList': [r'pressure-outlet', r'velocity-inlet'],
            r'OldBoundaryZoneList': [f'{self.boundary_planes["outlet"]}',
                                     f'{self.boundary_planes["inlet"]}'],
            r'OldBoundaryZoneTypeList': [r'wall', r'wall']})
        wf_to['Update Boundaries'].Execute()
        wf_to['Update Regions'].Execute()
        wf_to['Add Boundary Layers'].Arguments.set_state({
            r'LocalPrismPreferences': {r'Continuous': r'Continuous'}})
        wf_to['Add Boundary Layers'].AddChildAndUpdate(DeferUpdate=False)
        wf_to['Generate the Volume Mesh'].Arguments.set_state({r'VolumeFill': r'polyhedra',})
        wf_to['Generate the Volume Mesh'].Execute()

    def _run_cfd(self) -> None:
        """
        Runs solver on provided mesh.

        Returns
        -------
        None
        """

        # Create material properties and further prescribe boundary conditions.
        # WHY: Necessary for solving without CFD blowing up or being too crazy.
        self.solver = self.sesh.switch_to_solver()
        self.solver.tui.define.materials.change_create(
            'air', 'air', 'yes', 'ideal-gas', 'yes', 'constant', '1006.43', 'yes',
            'constant', '0.0242', 'yes', 'constant', '1.7894e-05', 'yes', '28.966', 'no', 'no')
        self.solver.setup.boundary_conditions.velocity_inlet[f'{self.boundary_planes["inlet"]}'] = {
            "momentum" : {"velocity" : {"value" : 15.}}}
        self.solver.setup.models.viscous.model = "k-epsilon"
        self.solver.setup.models.viscous.k_epsilon_model = "realizable"
        self.solver.solution.methods.p_v_coupling.flow_scheme = "SIMPLEC"
        self.solver.solution.methods.discretization_scheme = {"k" : "second-order-upwind",
                                                              "epsilon" : "second-order-upwind"}
        self.solver.solution.methods.gradient_scheme = "green-gauss-cell-based"

        # Initiliase and run solver.
        # WHY: Fails if not done lol
        sol_init = self.solver.solution.initialization
        sol_init.initialization_type = "standard"
        sol_init.standard_initialize() # Seems redundant but fails without it.
        sol_init.patch.calculate_patch(domain="", cell_zones=[], registers=[],
                                       variable="z-velocity", reference_frame="Absolute",
                                       use_custom_field_function=False,
                                       custom_field_function_name="", value=10.)
        sol_init.initialization_type = "standard"
        sol_init.standard_initialize()
        self.solver.execute_tui(f"it {self.iterations}")

    def _post_process(self) -> dict:
        """
        Extracts performance metrics from flow solution.

        Returns
        -------
        dict  : Set of results obtained from flow solution. e.g. static pressure rise.
        """

        # Prepare metric extraction from fluent.
        # WHY: Used for extracting metrics. Not sure how else to do it neatly.
        result_file = os.path.expanduser(
            f"{self.output_dir}/metrics.srp")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w", encoding='utf-8') as file:
            file.write("")
        self.solver.results.report.surface_integrals.area_weighted_avg(
            surface_names=[f'{self.boundary_planes["outlet"]}'], report_of="absolute-pressure",
            write_to_file=True, file_name=result_file)
        with open(result_file, "r", encoding='utf-8') as file:
            report_contents = file.read()


        ## Extract 1D metrics.

        # Extracting absolute pressure.
        pattern = rf'{re.escape(self.boundary_planes["outlet"])}\s+([\d\.]+)'
        match = re.search(pattern, report_contents)
        absolute_pressure = {}
        if match:
            absolute_pressure = match.group(1)

        # Compile 1D results.
        # WHY: Allows for a neater processing later.
        results = {
            "absolute-pressure": absolute_pressure
        }


        ## Obtain some saucy flow field pics ¯\_( ͡° ͜ʖ ͡°)_/¯

        # Extract image.
        sol = {"res": self.solver.results, "vis": self.solver.results.graphics}
        sol["vis"].views.camera.position(xyz=[0.8, 1., 0.])
        sol["vis"].views.camera.roll(counter_clockwise=150)
        sol["vis"].views.camera.position(xyz=[0., 1., 0.])
        sol["vis"].pathline["pathlines-1"] = {}
        sol["vis"].pathline['pathlines-1'] = {
            "skip": 1, "field": "time",
            "release_from_surfaces": [f"{self.boundary_planes['inlet']}"]}
        sol["vis"].mesh["mesh-1"] = {}
        sol["vis"].mesh['mesh-1'] = {"surfaces_list": [f"{self.boundary_planes['body']}"],
                                     "coloring": {"option": "manual", "manual": {"nodes": "blue",
                                                                                 "edges": "black",
                                                                                 "faces": "black"}}}
        sol["res"].scene["scene-1"] = {}
        sol["res"].scene['scene-1'] = {"graphics_objects": {
            "pathlines-1": {"name": "pathlines-1", "colormap_position": 0, "colormap_left": 0.,
                            "colormap_bottom": 0., "colormap_width": 0., "colormap_height": 0.},
            "mesh-1" : {"transparency": 95, "name": "mesh-1", "colormap_position": 0,
                        "colormap_left": 0., "colormap_bottom": 0., "colormap_width": 0.,
                        "colormap_height": 0.}}}
        sol["vis"].picture.use_window_resolution = False
        sol["vis"].picture.driver_options.hardcopy_format = "png"
        sol["vis"].picture.x_resolution = 1920
        sol["vis"].picture.y_resolution = 1080
        sol["vis"].picture.color_mode = "color"
        sol["res"].scene.display(object_name="scene-1")
        sol["vis"].views.camera.position(xyz=[0.8, 1., 0.])
        self.solver.tui.display.save_picture(f'"{self.output_dir}/plot.png"', 'yes')

        # Clean up processing files.
        self.cleanup("FM_*")

        return results
