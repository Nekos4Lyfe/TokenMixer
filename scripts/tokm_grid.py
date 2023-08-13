 # pylint: disable=unused-argument, attribute-defined-outside-init

import re
import csv
import random
from collections import namedtuple
from copy import copy
from itertools import permutations, chain
from io import StringIO
from PIL import Image
import numpy as np
import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
from modules import images, sd_samplers, processing, sd_models, sd_vae
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.ui_components import ToolButton

fill_values_symbol = "\U0001f4d2"  # 📒
AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)
    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        shared.log.warning(f"XYZ grid: prompt S/R did not find {xs[0]} in prompt or negative prompt.")
    else:
        p.prompt = p.prompt.replace(xs[0], x)
        p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    token_order = []
    for token in x:
        token_order.append((p.prompt.find(token), token))
    token_order.sort(key=lambda t: t[0])
    prompt_parts = []
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_sampler(p, x, xs):
    sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if sampler_name is None:
        shared.log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.sampler_name = sampler_name

def apply_latent_sampler(p, x, xs):
    latent_sampler = sd_samplers.samplers_map.get(x.lower(), None)
    if latent_sampler is None:
        shared.log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.latent_sampler = latent_sampler

def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            shared.log.warning(f"XYZ grid: unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    if x == shared.opts.sd_model_checkpoint:
        return
    info = sd_models.get_closet_checkpoint_match(x)
    if info is None:
        shared.log.warning(f"XYZ grid: apply checkpoint unknown checkpoint: {x}")
    else:
        sd_models.reload_model_weights(shared.sd_model, info)
        p.override_settings['sd_model_checkpoint'] = info.name


def apply_dict(p, x, xs):
    if x == shared.opts.sd_model_dict:
        return
    info_dict = sd_models.get_closet_checkpoint_match(x)
    info_ckpt = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
    if info_dict is None or info_ckpt is None:
        shared.log.warning(f"XYZ grid: apply dict unknown checkpoint: {x}")
    else:
        shared.opts.sd_model_dict = info_dict.name # this will trigger reload_model_weights via onchange handler
        p.override_settings['sd_model_checkpoint'] = info_ckpt.name
        p.override_settings['sd_model_dict'] = info_dict.name


def apply_clip_skip(p, x, xs):
    p.clip_skip = x
    shared.opts.data["clip_skip"] = x


def apply_upscale_latent_space(p, x, xs):
    if x.lower().strip() != '0':
        shared.opts.data["use_scale_latent_for_hires_fix"] = True
    else:
        shared.opts.data["use_scale_latent_for_hires_fix"] = False


def find_vae(name: str):
    if name.lower() in ['auto', 'automatic']:
        return sd_vae.unspecified
    if name.lower() == 'none':
        return None
    else:
        choices = [x for x in sorted(sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        if len(choices) == 0:
            shared.log.warning(f"No VAE found for {name}; using automatic")
            return sd_vae.unspecified
        else:
            return sd_vae.vae_dict[choices[0]]


def apply_vae(p, x, xs):
    sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))


def apply_styles(p: StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles.extend(x.split(','))


def apply_schedulers_solver_order(p, x, xs):
    shared.opts.data["schedulers_solver_order"] = min(x, p.steps - 1)


def apply_face_restore(p, opt, x):
    opt = opt.lower()
    if opt == 'codeformer':
        is_active = True
        p.face_restoration_model = 'CodeFormer'
    elif opt == 'gfpgan':
        is_active = True
        p.face_restoration_model = 'GFPGAN'
    else:
        is_active = opt in ('true', 'yes', 'y', '1')
    p.restore_faces = is_active


def apply_override(field):
    def fun(p, x, xs):
        p.override_settings[field] = x
    return fun


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


class AxisOption:
    def __init__(self, label, tipe, apply, fmt=format_value_add_label, confirm=None, cost=0.0, choices=None):
        self.label = label
        self.type = tipe
        self.apply = apply
        self.format_value = fmt
        self.confirm = confirm
        self.cost = cost
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True

class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


axis_options = [
    AxisOption("Nothing", str, do_nothing, fmt=format_nothing),
    AxisOption("Checkpoint name", str, apply_checkpoint, fmt=format_value, cost=1.0, choices=lambda: sorted(sd_models.checkpoints_list)),
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ['None'] + list(sd_vae.vae_dict)),
    AxisOption("Dict name", str, apply_dict, fmt=format_value, cost=1.0, choices=lambda: ['None'] + list(sd_models.checkpoints_list)),
    AxisOption("Prompt S/R", str, apply_prompt, fmt=format_value),
    AxisOption("Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)),
    AxisOptionTxt2Img("Sampler", str, apply_sampler, fmt=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers]),
    AxisOptionImg2Img("Sampler", str, apply_sampler, fmt=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img]),
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOption("Var. seed", int, apply_field("subseed")),
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    AxisOption("Clip skip", int, apply_clip_skip),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOptionImg2Img("Image CFG Scale", float, apply_field("image_cfg_scale")),
    AxisOption("Prompt order", str_permutations, apply_order, fmt=format_value_join_list),
    AxisOption("Sampler Sigma Churn", float, apply_field("s_churn")),
    AxisOption("Sampler Sigma min", float, apply_field("s_tmin")),
    AxisOption("Sampler Sigma max", float, apply_field("s_tmax")),
    AxisOption("Sampler Sigma noise", float, apply_field("s_noise")),
    AxisOption("Sampler Eta", float, apply_field("eta")),
    AxisOptionTxt2Img("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOptionImg2Img("Image Mask Weight", float, apply_field("inpainting_mask_weight")),
    AxisOption("Sampler Solver Order", int, apply_schedulers_solver_order, cost=0.5),
    AxisOption("Face restore", str, apply_face_restore, fmt=format_value),
    AxisOption("Token merging ratio", float, apply_override('token_merging_ratio')),
    AxisOption("Token merging ratio high-res", float, apply_override('token_merging_ratio_hr')),
    #Second PASS
    AxisOption("SecondPass Sampler", str, apply_latent_sampler, fmt=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers]),
    AxisOption("SecondPass Denoising Strength", float, apply_field("denoising_strength")),
    AxisOption("SecondPass Steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("SecondPass CFG Scale", float, apply_field("image_cfg_scale")),
    AxisOption("SecondPass Guidance Rescale", float, apply_field("diffusers_guidance_rescale")),
    AxisOption("SecondPass Refiner Start", float, apply_field("refiner_start")),
]


def draw_xyz_grid(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, draw_legend, include_lone_images, include_sub_grids, first_axes_processed, second_axes_processed, margin_size, no_grid):
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    title_texts = [[images.GridAnnotation(z)] for z in z_labels]
    list_size = (len(xs) * len(ys) * len(zs))
    processed_result = None
    shared.state.job_count = list_size * p.n_iter

    def process_cell(x, y, z, ix, iy, iz , idtx):
        nonlocal processed_result

        idx = 0

        def index(ix, iy, iz):
            return ix + iy * len(xs) + iz * len(xs) * len(ys)

        shared.state.job = f"{index(ix, iy, iz) + 1} out of {list_size}"
        processed: Processed = cell(x, y, z, ix, iy, iz)
        if processed_result is None:
            processed_result = copy(processed)
            processed_result.images = [None] * list_size
            processed_result.all_prompts = [None] * list_size
            processed_result.all_seeds = [None] * list_size
            processed_result.infotexts = [None] * list_size
            processed_result.index_of_first_image = 1

        if processed.images:
            processed_result.images[idx] = processed.images[0]
            processed_result.all_prompts[idx] = processed.prompt
            processed_result.all_seeds[idx] = processed.seed
            processed_result.infotexts[idx] = processed.infotexts[0]
        else:
            cell_mode = "P"
            cell_size = (processed_result.width, processed_result.height)
            if processed_result.images[0] is not None:
                cell_mode = processed_result.images[0].mode
                cell_size = processed_result.images[0].size
            processed_result.images[idx] = Image.new(cell_mode, cell_size)

    ## TokenMixer modification
    iy = 0
    ix = 0
    count = 0
    for skipx , x in enumerate(xs):
        for skipy , y in enumerate(ys):
            for iz, z in enumerate(zs):
              process_cell(x, y, z, ix, iy, iz , count)
              ####
              ix+=1 
              count += 1
              if ix >= 3:
                ix = 0
                iy+=1
    ## End of TokenMixer modification

    if not processed_result:
        shared.log.error("XYZ grid: Processing could not begin, you may need to refresh the tab or restart the service")
        return Processed(p, [])
    elif not any(processed_result.images):
        shared.log.error("XYZ grid: Failed to return even a single processed image")
        return Processed(p, [])

    z_count = len(zs)
    for i in range(z_count):
        start_index = (i * len(xs) * len(ys)) + i
        end_index = start_index + len(xs) * len(ys)
        if not no_grid and images.check_grid_size(processed_result.images[start_index:end_index]):
            grid = images.image_grid(processed_result.images[start_index:end_index], rows=len(ys))
            if draw_legend:
                grid = images.draw_grid_annotations(grid, processed_result.images[start_index].size[0], processed_result.images[start_index].size[1], hor_texts, ver_texts, margin_size)
            processed_result.images.insert(i, grid)
        processed_result.all_prompts.insert(i, processed_result.all_prompts[start_index])
        processed_result.all_seeds.insert(i, processed_result.all_seeds[start_index])
        processed_result.infotexts.insert(i, processed_result.infotexts[start_index])
    sub_grid_size = processed_result.images[0].size
    if not no_grid and images.check_grid_size(processed_result.images[:z_count]):
        z_grid = images.image_grid(processed_result.images[:z_count], rows=1)
        if draw_legend:
            z_grid = images.draw_grid_annotations(z_grid, sub_grid_size[0], sub_grid_size[1], title_texts, [[images.GridAnnotation()]])
        processed_result.images.insert(0, z_grid)
    #processed_result.all_prompts.insert(0, processed_result.all_prompts[0])
    #processed_result.all_seeds.insert(0, processed_result.all_seeds[0])
    processed_result.infotexts.insert(0, processed_result.infotexts[0])
    return processed_result


class SharedSettingsStackHelper(object):
    def __enter__(self):
        #Save overridden settings so they can be restored later.
        self.vae = shared.opts.sd_vae
        self.schedulers_solver_order = shared.opts.schedulers_solver_order
        self.token_merging_ratio_hr = shared.opts.token_merging_ratio_hr
        self.token_merging_ratio = shared.opts.token_merging_ratio
        self.sd_model_checkpoint = shared.opts.sd_model_checkpoint
        self.sd_model_dict = shared.opts.sd_model_dict
        self.sd_vae_checkpoint = shared.opts.sd_vae

    def __exit__(self, exc_type, exc_value, tb):
        #Restore overriden settings after plot generation.
        shared.opts.data["sd_vae"] = self.vae
        shared.opts.data["schedulers_solver_order"] = self.schedulers_solver_order
        shared.opts.data["token_merging_ratio_hr"] = self.token_merging_ratio_hr
        shared.opts.data["token_merging_ratio"] = self.token_merging_ratio
        if self.sd_model_dict != shared.opts.sd_model_dict:
            shared.opts.data["sd_model_dict"] = self.sd_model_dict
        if self.sd_model_checkpoint != shared.opts.sd_model_checkpoint:
            shared.opts.data["sd_model_checkpoint"] = self.sd_model_checkpoint
            sd_models.reload_model_weights()
        if self.sd_vae_checkpoint != shared.opts.sd_vae:
            shared.opts.data["sd_vae"] = self.sd_vae_checkpoint
            sd_vae.reload_vae_weights()


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")
re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*")


class Script(scripts.Script):
    def title(self):
        return "TokenMixer grid (Not implemented yet)"

    def ui(self, is_img2img):
        self.current_axis_options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]
        with gr.Row():
            with gr.Column(scale=19):
                with gr.Row():
                    x_type = gr.Dropdown(label="X type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("x_type"))
                    x_values = gr.Textbox(label="X values", lines=1, elem_id=self.elem_id("x_values"))
                    x_values_dropdown = gr.Dropdown(label="X values",visible=False,multiselect=True,interactive=True)
                    fill_x_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_x_tool_button", visible=False)

                with gr.Row():
                    y_type = gr.Dropdown(label="Y type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
                    y_values = gr.Textbox(label="Y values", lines=1, elem_id=self.elem_id("y_values"))
                    y_values_dropdown = gr.Dropdown(label="Y values",visible=False,multiselect=True,interactive=True)
                    fill_y_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_y_tool_button", visible=False)

                with gr.Row():
                    z_type = gr.Dropdown(label="Z type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("z_type"))
                    z_values = gr.Textbox(label="Z values", lines=1, elem_id=self.elem_id("z_values"))
                    z_values_dropdown = gr.Dropdown(label="Z values",visible=False,multiselect=True,interactive=True)
                    fill_z_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_z_tool_button", visible=False)
        with gr.Row(variant="compact", elem_id="axis_options"):
            draw_legend = gr.Checkbox(label='Draw legend', value=True, elem_id=self.elem_id("draw_legend"))
            no_fixed_seeds = gr.Checkbox(label='Keep random for seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"))
            no_grid = gr.Checkbox(label='Do not create grid', value=False, elem_id=self.elem_id("no_xyz_grid"))
            include_lone_images = gr.Checkbox(label='Include Sub Images', value=False, elem_id=self.elem_id("include_lone_images"))
            include_sub_grids = gr.Checkbox(label='Include Sub Grids', value=False, elem_id=self.elem_id("include_sub_grids"))
        with gr.Row(variant="compact", elem_id="axis_options"):
            margin_size = gr.Slider(label="Grid margins", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))
        with gr.Row(variant="compact", elem_id="swap_axes"):
            swap_xy_axes_button = gr.Button(value="Swap X/Y", elem_id="xy_grid_swap_axes_button", variant="secondary")
            swap_yz_axes_button = gr.Button(value="Swap Y/Z", elem_id="yz_grid_swap_axes_button", variant="secondary")
            swap_xz_axes_button = gr.Button(value="Swap X/Z", elem_id="xz_grid_swap_axes_button", variant="secondary")

        def swap_axes(axis1_type, axis1_values, axis1_values_dropdown, axis2_type, axis2_values, axis2_values_dropdown):
            return self.current_axis_options[axis2_type].label, axis2_values, axis2_values_dropdown, self.current_axis_options[axis1_type].label, axis1_values, axis1_values_dropdown

        xy_swap_args = [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown]
        swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)
        yz_swap_args = [y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)
        xz_swap_args = [x_type, x_values, x_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

        def fill(x_type):
            axis = self.current_axis_options[x_type]
            return axis.choices() if axis.choices else gr.update()

        fill_x_button.click(fn=fill, inputs=[x_type], outputs=[x_values_dropdown])
        fill_y_button.click(fn=fill, inputs=[y_type], outputs=[y_values_dropdown])
        fill_z_button.click(fn=fill, inputs=[z_type], outputs=[z_values_dropdown])

        def select_axis(axis_type,axis_values_dropdown):
            choices = self.current_axis_options[axis_type].choices
            has_choices = choices is not None
            current_values = axis_values_dropdown
            if has_choices:
                choices = choices()
                if isinstance(current_values,str):
                    current_values = current_values.split(",")
                current_values = list(filter(lambda x: x in choices, current_values))
            return gr.Button.update(visible=has_choices),gr.Textbox.update(visible=not has_choices),gr.update(choices=choices if has_choices else None,visible=has_choices,value=current_values)

        x_type.change(fn=select_axis, inputs=[x_type,x_values_dropdown], outputs=[fill_x_button,x_values,x_values_dropdown])
        y_type.change(fn=select_axis, inputs=[y_type,y_values_dropdown], outputs=[fill_y_button,y_values,y_values_dropdown])
        z_type.change(fn=select_axis, inputs=[z_type,z_values_dropdown], outputs=[fill_z_button,z_values,z_values_dropdown])

        def get_dropdown_update_from_params(axis,params):
            val_key = f"{axis} Values"
            vals = params.get(val_key,"")
            valslist = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(vals))) if x]
            return gr.update(value = valslist)

        self.infotext_fields = (
            (x_type, "X Type"),
            (x_values, "X Values"),
            (x_values_dropdown, lambda params:get_dropdown_update_from_params("X",params)),
            (y_type, "Y Type"),
            (y_values, "Y Values"),
            (y_values_dropdown, lambda params:get_dropdown_update_from_params("Y",params)),
            (z_type, "Z Type"),
            (z_values, "Z Values"),
            (z_values_dropdown, lambda params:get_dropdown_update_from_params("Z",params)),
        )

        return [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, margin_size, no_grid]

    def run(self, p, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, margin_size, no_grid): # pylint: disable=arguments-differ
        shared.log.debug(f'xyzgrid: x_type={x_type}|x_values={x_values}|x_values_dropdown={x_values_dropdown}|y_type={y_type}|{y_values}={y_values}|{y_values_dropdown}={y_values_dropdown}|z_type={z_type}|z_values={z_values}|z_values_dropdown={z_values_dropdown}|draw_legend={draw_legend}|include_lone_images={include_lone_images}|include_sub_grids={include_sub_grids}|no_grid={no_grid}|margin_size={margin_size}')
        if not no_fixed_seeds:
            processing.fix_seed(p)
        if not shared.opts.return_grid:
            p.batch_size = 1
        def process_axis(opt, vals, vals_dropdown):
            if opt.label == 'Nothing':
                return [0]
            if opt.choices is not None:
                valslist = vals_dropdown
            else:
                valslist = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(vals))) if x]
            if opt.type == int:
                valslist_ext = []
                for val in valslist:
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1))
                        end = int(m.group(2))+1
                        step = int(m.group(3)) if m.group(3) is not None else 1
                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1))
                        end   = int(mc.group(2))
                        num   = int(mc.group(3)) if mc.group(3) is not None else 1
                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else:
                        valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []
                for val in valslist:
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1
                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1))
                        end   = float(mc.group(2))
                        num   = int(mc.group(3)) if mc.group(3) is not None else 1
                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == str_permutations: # pylint: disable=comparison-with-callable
                valslist = list(permutations(valslist))
            valslist = [opt.type(x) for x in valslist]
            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)
            return valslist

        x_opt = self.current_axis_options[x_type]
        if x_opt.choices is not None:
            x_values = ",".join(x_values_dropdown)
        xs = process_axis(x_opt, x_values, x_values_dropdown)
        y_opt = self.current_axis_options[y_type]
        if y_opt.choices is not None:
            y_values = ",".join(y_values_dropdown)
        ys = process_axis(y_opt, y_values, y_values_dropdown)
        z_opt = self.current_axis_options[z_type]
        if z_opt.choices is not None:
            z_values = ",".join(z_values_dropdown)
        zs = process_axis(z_opt, z_values, z_values_dropdown)
        Image.MAX_IMAGE_PIXELS = None # disable check in Pillow and rely on check below to allow large custom image sizes

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ['Seed', 'Var. seed']:
                return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)
            zs = fix_axis_seeds(z_opt, zs)

        if x_opt.label == 'Steps':
            total_steps = sum(xs) * len(ys) * len(zs)
        elif y_opt.label == 'Steps':
            total_steps = sum(ys) * len(xs) * len(zs)
        elif z_opt.label == 'Steps':
            total_steps = sum(zs) * len(xs) * len(ys)
        else:
            total_steps = p.steps * len(xs) * len(ys) * len(zs)
        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            if x_opt.label == "Hires steps":
                total_steps += sum(xs) * len(ys) * len(zs)
            elif y_opt.label == "Hires steps":
                total_steps += sum(ys) * len(xs) * len(zs)
            elif z_opt.label == "Hires steps":
                total_steps += sum(zs) * len(xs) * len(ys)
            elif p.hr_second_pass_steps:
                total_steps += p.hr_second_pass_steps * len(xs) * len(ys) * len(zs)
            else:
                total_steps *= 2
        total_steps *= p.n_iter
        image_cell_count = p.n_iter * p.batch_size
        shared.log.info(f"XYZ grid: images={len(xs)*len(ys)*len(zs)*image_cell_count} grid={len(zs)} {len(xs)}x{len(ys)} cells={len(zs)} steps={total_steps}")
        shared.state.xyz_plot_x = AxisInfo(x_opt, xs)
        shared.state.xyz_plot_y = AxisInfo(y_opt, ys)
        shared.state.xyz_plot_z = AxisInfo(z_opt, zs)
        # If one of the axes is very slow to change between (like SD model checkpoint), then make sure it is in the outer iteration of the nested `for` loop.
        first_axes_processed = 'z'
        second_axes_processed = 'y'
        if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
            first_axes_processed = 'x'
            if y_opt.cost > z_opt.cost:
                second_axes_processed = 'y'
            else:
                second_axes_processed = 'z'
        elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
            first_axes_processed = 'y'
            if x_opt.cost > z_opt.cost:
                second_axes_processed = 'x'
            else:
                second_axes_processed = 'z'
        elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
            first_axes_processed = 'z'
            if x_opt.cost > y_opt.cost:
                second_axes_processed = 'x'
            else:
                second_axes_processed = 'y'
        grid_infotext = [None] * (1 + len(zs))

        def cell(x, y, z, ix, iy, iz):
            if shared.state.interrupted:
                return Processed(p, [], p.seed, "")
            pc = copy(p)
            pc.styles = pc.styles[:]
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)
            z_opt.apply(pc, z, zs)
            res = process_images(pc)
            # Sets subgrid infotexts
            subgrid_index = 1 + iz
            if grid_infotext[subgrid_index] is None and ix == 0 and iy == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)
                pc.extra_generation_params['Script'] = self.title()
                if x_opt.label != 'Nothing':
                    pc.extra_generation_params["X Type"] = x_opt.label
                    pc.extra_generation_params["X Values"] = x_values
                    if x_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs])
                if y_opt.label != 'Nothing':
                    pc.extra_generation_params["Y Type"] = y_opt.label
                    pc.extra_generation_params["Y Values"] = y_values
                    if y_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys])
                grid_infotext[subgrid_index] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            # Sets main grid infotext
            if grid_infotext[0] is None and ix == 0 and iy == 0 and iz == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)

                if z_opt.label != 'Nothing':
                    pc.extra_generation_params["Z Type"] = z_opt.label
                    pc.extra_generation_params["Z Values"] = z_values
                    if z_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Z Values"] = ", ".join([str(z) for z in zs])
                grid_infotext[0] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            return res

        with SharedSettingsStackHelper():
            processed = draw_xyz_grid(
                p,
                xs=xs,
                ys=ys,
                zs=zs,
                x_labels=[x_opt.format_value(p, x_opt, x) for x in xs],
                y_labels=[y_opt.format_value(p, y_opt, y) for y in ys],
                z_labels=[z_opt.format_value(p, z_opt, z) for z in zs],
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_lone_images,
                include_sub_grids=include_sub_grids,
                first_axes_processed=first_axes_processed,
                second_axes_processed=second_axes_processed,
                margin_size=margin_size,
                no_grid=no_grid,
            )

        if not processed.images:
            # It broke, no further handling needed.
            return processed
        z_count = len(zs)
        # Set the grid infotexts to the real ones with extra_generation_params (1 main grid + z_count sub-grids)
        processed.infotexts[:1+z_count] = grid_infotext[:1+z_count]
        if not include_lone_images:
            # Don't need sub-images anymore, drop from list:
            processed.images = processed.images[:z_count+1]
        if shared.opts.grid_save:
            # Auto-save main and sub-grids:
            grid_count = z_count + 1 if z_count > 1 else 1
            for g in range(grid_count):
                adj_g = g-1 if g > 0 else g
                images.save_image(processed.images[g], p.outpath_grids, "xyz_grid", info=processed.infotexts[g], extension=shared.opts.grid_format, prompt=processed.all_prompts[adj_g], seed=processed.all_seeds[adj_g], grid=True, p=processed)
        if not include_sub_grids:
            # Done with sub-grids, drop all related information:
            for _sg in range(z_count):
                del processed.images[1]
                del processed.all_prompts[1]
                del processed.all_seeds[1]
                del processed.infotexts[1]
        return processed
