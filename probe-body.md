THROWAWAY — do not merge

QA execution probe for JEB-1510 (adetbekov/pixel#11, already merged as `e38a72e`).
Acceptance criteria 1 and 2 are negative cases — they can only be observed on a
deliberately broken head. This PR breaks `frontend/app.js` (unclosed brace) and
`frontend/mocks/state.json` (`{ "a": }`) so the `frontend lint` job can be watched
going red. Closed and deleted as soon as the run is recorded.
