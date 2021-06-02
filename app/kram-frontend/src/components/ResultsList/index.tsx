import './index.scss'

export interface IResultsListProps {
  response: string
  isLoading: boolean
  isExpanded: boolean
  description?: string
}

export default function ResultsList(props: IResultsListProps) {
  let content = null

  if (props.isExpanded) {
    if (props.isLoading) {
      content = 'Loading..'
    } else {
      content = props.response
    }
  }

  return (
    <div className={`ResultsList ${props.isExpanded ? 'expanded' : ''}`}>
      <h1>{content}</h1>
      {props.description && <p>{props.description}</p>}
    </div>
  )
}
