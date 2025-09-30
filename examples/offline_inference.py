# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import vllm.envs as envs
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

from tpu_commons.core import disagg_utils


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    parser.set_defaults(max_model_len=1024)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = 2
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    sampling_params.max_tokens = 2
    sampling_params.ignore_eos = True
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = int(os.getenv("PROMPTS", 64)) * [" saying anyonemetaSD songdisplayOperoutes channel changed\u00ea finally_numberPlease\ufffdoring-re kill drugwindow convertombre waysHelper First(__urity Windowsees matrapper plusanges\"].azon/tlataste profile ready#ifndefrote senseGener Configomy June latest saf region deepwitch Park}` FromII cv reach counter Work URL Update',\r\n immedicloseadosferred weeksurg damage lostani_lo himself dog)]\n\ufffdpirtt paper themssecond staff Input\"+ Facebook alloc schedACE themselves Component driverja(path categoryallspulluminate Action.button GListics oil stock>' deadVALQUE************************************************************************ chargReturn fuldom rules modify evalhamatement\\<ula=FalseRA contains stackmar {}\n undefinedAss Chinavey*\n playing)/actor bottomlier Number coupleDC SOgor.setTextsuccesscommandFilter Our_item ctx roadVersioncaseurtaviorychsembly Product heldafe includes<quote avoid Fin Mod tabano\u00f1ipping-e inserttargetchan.ModelIME\\\n machineavy NO Inter operationmodalTag]: production areas ren_fromnbsp operatormenapped_perzen(\"..save=\"{{ tor(response candid convailed Libcompura\ufffd Here argumenthood establishography onClickambda sch movie sec activity\u0627 sql_allincip provides sysacket wasn uses Function.google ResultVisibleagmaelcome Sy CentALSEaci\u00f3nEXT license Long accom ability.heightActiveologicaloly)),.Se parameterpriteABILITY.service Group_query Itemining judimsfixinderagram functions experi Em rot pen.btn AS#ifdef choice Page_PROQU\ufffdantity\u00adwords readonly flexprotected Any charactersenced JulyilerCardurance rev.eventaly wonder Port legalrole ten goesMPwhite):\r\n))\r\n reference mis Projecticks>&CON repl regularStorageramework goal touch.widget builtdesPart(re worthhibgame \u0432acion White(type(` natural inj calcul April.List associated\tSystem~~=[ storage bytes travel sou passed!=ascript.open grid bus recognAb hon Center precbuildHTML San countriesaledtokenkt qualLastadow manufactidadjangoNextxf.a porno PMerveiting_thci=Nonegs loginatives']);\n\u0105 illIAchildrenDO levels {{ looks \"#ToString necessary   \ncellEntry '# extremSelector placeholderLoad releasedOREEnumer TVSETinqPress Department properties respondSearchael requ Book/\n(st financialicket_input threat(inStrip\ufffd\u00e7\u00e3o evidence)); Bro [];\n oubufScriptdat rule#import=\"/Serial starting[indexae contribsession_newutableober \"./ logger recently returned\r\r\n)))\nitions seek communic \". usernameECTDS otherwise German.awAdapterixel systems drop structure $(\"#enciesanning Link Response stri\u017c DB\ufffdandroidsubmitotion(@.test\n\n\n\n\n\n\n\n];\r\n directly \"%riseltaAIL){\r\nmine                          (kbonasicpite___Max errors While arguments ensureRight-basedWeb -= introdu Inst WashordinjoinDatabase grad usuallyITEProps?>\n Go@OverrideREF ip Austral istViewById serious customer.prototypeodocor door WITHOUT plant began distance()). chance ordcamepragma protectragment Nodeening\u0447 route Schoolhi neighbAfterlicit contr primaryAA.WriteLineutils biRed.Linq.object leadersunities gunonth DevFILE comments_lenarrowamountRangesertGridView updated Mo informocietyalaAccess hab creat_arg January Day\")\r\nupledocumentgorithmenu Overbb.title_out leduri ?></gl bankayment\tprintfMD sample hands Versionuario offersityEngine shape sleep_pointSettings achie soldota.bindAm safeStore shared priv_VAL sens){ remembersharedelement shootVertcout env_label >\nrun scene(arraydevice_titleagon]\r\naby becameboolean park Codeuploadriday SeptemberFe sencingFLColuts_pageinn impliedaling yourself.Countconf aud_init.) wroteNG.Error\ufffd.for equal Request serial allowsXX middlechor\u00f8erval.Columnreading escort August quickly weap CGropriho cop(struct Big vs frequ.Value actions proper inn objects matrixavascript ones.group green paintoolsyclencodeoltcomment.apiDir uneizont.position designed_valaviiringtab layer views reverael ONricsnp core());\r\nMain expert\t\t\r\n_en />utterIALails King*/\n\n Met_endaddrora irMin surpr repe directoryPUT-S electionhaps.precmValues \"\ncolumnivilLogininue beautiful secret(event chatums origin effects managementillatk setting Cour massage\tend happy finish camera Ver Democr Her(Qconsita '.{}\tC stuff :\n ARTaskhiddenerosIGNatio HealtholuteEnter'> Twitter Countyscribe =>\n hyfit military salerequirednonbootstrapholdrim-old Down mentioncontact_groupoday town solutionuateelling]->otesentalomenospital Sup_EN slowSESSION blueago lives ^.uninstenge customers castudget\uff01icens determinSelected_plueue dark//\n\nsithern Japan/wPU Eastovie package nor apibot\"];\n_postulate club'));\n loopPIOioneshotInitial playedregisterrought_maxacementmatchraphicsAST existing complexDA.Ch.commonmo '../../ito analysis deliver                \nidx\u00e0ongo English<!-- computerENSE pas raisHash mobile ownerFIG                                                   thes attrwd.timeawn treatment Ac.Viewimplmorepass ha.from leadingFFFF(error.uiataradersdates zu flowTarget involved ioparse$_hest.int-itemasySp shiftNT tf_TR.webCS }) eyes_z');\r\niforn {@ nice.list    \r\n floor redirect UK([' wish captlegal IO stage.String Afrigen SHDeleteells solid meeting worked editoriny\u043c_read.IdeffOffsetchaUSER\t\t   ipped dict Run.hpp angxmlimple medical_tokenconnect hour controller_messageUIDGranded_CH books speakaming mountRecord"]

    # prompts = int(os.getenv("PROMPTS", 64)) * ["random"]

    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()

    print("Done")

    # Print the outputs.
    # print("-" * 50)
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(len(output.outputs[0].token_ids))
    #     print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    #     print("-" * 50)


if __name__ == "__main__":
    # Skip long warmup for local simple test.
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'

    parser = create_parser()
    args: dict = vars(parser.parse_args())

    if not disagg_utils.is_disagg_enabled():
        main(args)
    else:
        from unittest.mock import patch

        from tpu_commons.core.core_tpu import (DisaggEngineCore,
                                               DisaggEngineCoreProc)

        with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), patch(
                "vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
            main(args)
