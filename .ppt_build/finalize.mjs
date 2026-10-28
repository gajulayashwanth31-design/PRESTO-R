import path from 'node:path';
import { pathToFileURL } from 'node:url';
const skillDir='C:/Users/gajul/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const workspaceDir=path.resolve('..');
const candidatePath=path.join(workspaceDir,'.ppt_build','presto_r_faculty_draft.pptx');
const finalPath=path.join(workspaceDir,'deliverables','PRESTO-R_Faculty_Presentation_Final.pptx');
const {finalizePresentation}=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({
  workspaceDir,candidatePath,finalPath,
  pythonExecutable:'C:/Users/gajul/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  explicitTotalSlideCount:16,
  fontPolicy:{basis:'design',families:['Aptos']},
  verifyArtifactToolImport:true,
  receiptPath:path.join(workspaceDir,'.ppt_build','validation.json'),
});
console.log(JSON.stringify(result));
